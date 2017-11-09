"""
Functions for dealing with data input and output.

"""

import os
import gzip
import logging
import numpy as np
import struct

logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------#
#                            GENERAL I/O FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def smart_open(filename, mode=None):
    """Opens a file normally or using gzip based on the extension."""
    if os.path.splitext(filename)[-1] == ".gz":
        if mode is None:
            mode = "rb"
        return gzip.open(filename, mode)
    else:
        if mode is None:
         mode = "r"
        return open(filename, mode)

def read_kaldi_ark_from_scp(uid, offset, batch_size, buffer_size, scp_fn, ark_base_dir=""):
    """
    Read a binary Kaldi archive and return a dict of Numpy matrices, with the
    utterance IDs of the SCP as keys. Based on the code:
    https://github.com/yajiemiao/pdnn/blob/master/io_func/kaldi_feat.py

    Parameters
    ----------
    ark_base_dir : str
        The base directory for the archives to which the SCP points.
    """

    ark_dict = {}
    totframes = 0
    lines = 0
    with open(scp_fn) as f:
        for line in f:
            lines = lines + 1
            if lines<=uid:
                continue
            if line == "":
                continue
            utt_id, path_pos = line.replace("\n", "").split()
            ark_path, pos = path_pos.split(":")
            if 'bagchid' in ark_path:
                ark_path_list = ark_path.split('/')
                ark_path = os.path.join(*ark_path_list[-3:])
            # print(ark_base_dir, ark_path)
            ark_path = os.path.join(ark_base_dir, ark_path)
            ark_read_buffer = smart_open(ark_path, "rb")
            ark_read_buffer.seek(int(pos),0)
            header = struct.unpack("<xcccc", ark_read_buffer.read(5))
            #assert header[0] == "B", "Input .ark file is not binary"
            rows = 0
            cols = 0
            m,rows = struct.unpack("<bi", ark_read_buffer.read(5))
            n,cols = struct.unpack("<bi", ark_read_buffer.read(5))
            tmp_mat = np.frombuffer(ark_read_buffer.read(rows*cols*4), dtype=np.float32)
            if len(tmp_mat) != rows * cols:
                return {}, lines
            utt_mat = np.reshape(tmp_mat, (rows, cols))
            #utt_mat_list=utt_mat.tolist()
            ark_read_buffer.close()
            ark_dict[utt_id] = utt_mat
            totframes += rows
            if totframes>=(batch_size*buffer_size-offset):
                break

    return ark_dict,lines

def kaldi_write_mats(ark_path, utt_id, utt_mat):
    ark_write_buf = smart_open(ark_path, "ab")
    utt_mat = np.asarray(utt_mat, dtype=np.float32)
    batch, rows, cols = utt_mat.shape
    ark_write_buf.write(struct.pack('<%ds'%(len(utt_id)), utt_id))
    ark_write_buf.write(struct.pack('<cxcccc', b' ',b'B',b'F',b'M',b' '))
    ark_write_buf.write(struct.pack('<bi', 4, rows))
    ark_write_buf.write(struct.pack('<bi', 4, cols))
    ark_write_buf.write(utt_mat)


class DataLoader:
    """ Class for loading features and labels from file into a buffer, and batching. """

    def __init__(self,
            base_dir,
            in_frame_file,
            out_frame_file,
            batch_size,
            buffer_size,
            context,
            out_frame_count,
            shuffle):

        """ Initialize the data loader including filling the buffer """
        self.data_dir = base_dir
        self.in_frame_file = in_frame_file
        self.out_frame_file = out_frame_file
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.context = context
        self.out_frame_count = out_frame_count
        self.shuffle = shuffle
        
        self.uid = 0
        self.offset = 0

    def read_mats(self, frame_file):
        """ Read features from file into a buffer """
        #Read a buffer containing buffer_size*batch_size+offset
        #Returns a line number of the scp file
        scp_fn = os.path.join(self.data_dir, frame_file)
        ark_dict, uid = read_kaldi_ark_from_scp(
                self.uid,
                self.offset,
                self.batch_size,
                self.buffer_size,
                scp_fn,
                self.data_dir)

        return ark_dict, uid

    def _fill_buffer(self):
        """ Read data from files into buffers """

        # Read data
        in_frame_dict, uid_new  = self.read_mats(self.in_frame_file)
        out_frame_dict, uid_new = self.read_mats(self.out_frame_file)

        if len(in_frame_dict) == 0:
            self.empty = True
            return

        self.uid = uid_new

        ids = sorted(in_frame_dict.keys())

        if not hasattr(self, 'offset_in_frames'):
            self.offset_in_frames = np.empty((0, in_frame_dict[ids[0]].shape[1]), np.float32)

        if not hasattr(self, 'offset_out_frames'):
            self.offset_out_frames = np.empty((0, out_frame_dict[ids[0]].shape[1]), np.float32)

        # Create frame buffers
        in_frames = [in_frame_dict[i] for i in ids]
        in_frames = np.vstack(in_frames)
        in_frames = np.concatenate((self.offset_in_frames, in_frames), axis=0)

        out_frames = [out_frame_dict[i] for i in ids]
        out_frames = np.vstack(out_frames)
        out_frames = np.concatenate((self.offset_out_frames, out_frames), axis=0)

        # Put one batch into the offset frames
        cutoff = self.batch_size * self.buffer_size
        if in_frames.shape[0] >= cutoff:
            self.offset_in_frames = in_frames[cutoff:]
            in_frames = in_frames[:cutoff]
            self.offset_out_frames = out_frames[cutoff:]
            out_frames = out_frames[:cutoff]
            
            self.offset = self.offset_in_frames.shape[0]

        in_frames = np.pad(
            array     = in_frames,
            pad_width = ((self.context + self.out_frame_count // 2,),(0,)),
            mode      = 'edge')

        # Generate a random permutation of indexes
        if self.shuffle:
            self.indexes = np.random.permutation(out_frames.shape[0])
        else:
            self.indexes = np.arange(out_frames.shape[0])

        self.in_frame_buffer = in_frames
        self.out_frame_buffer = out_frames

    def batchify(self, include_deltas=True):
        """ Make a batch of frames and senones """

        batch_index = 0
        self.reset()

        while not self.empty:
            start = batch_index * self.batch_size
            end = min((batch_index+1) * self.batch_size, self.out_frame_buffer.shape[0])

            # Collect the data 
            in_frame_batch = np.stack((self.in_frame_buffer[i:i+self.out_frame_count+2*self.context,]
                for i in self.indexes[start:end]), axis = 0)

            out_frame_batch = np.stack((self.out_frame_buffer[i:i+self.out_frame_count,]
                for i in self.indexes[start:end]), axis = 0).squeeze()

            # Increment batch, and if necessary re-fill buffer
            batch_index += 1
            if batch_index * self.batch_size >= self.out_frame_buffer.shape[0]:
                batch_index = 0
                self._fill_buffer()

            if include_deltas:
                yield in_frame_batch, out_frame_batch
            else:
                yield in_frame_batch[:,:,:257], out_frame_batch


    def reset(self):
        self.uid = 0
        self.offset = 0
        self.empty = False

        self._fill_buffer()
