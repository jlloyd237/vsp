# -*- coding: utf-8 -*-
"""Processor classes for capturing images or video frames from camera or file, and processing them using a
dataflow pipeline of functions/transformations.
"""

import logging
import sys
import queue
from abc import ABC, abstractmethod
from threading import Thread

import numpy as np

from vsp.dataflow import DataflowQueueMT, DataflowFunctionMT, DataflowQueueMP, DataflowFunctionMP
from vsp.utils import compose


logging.basicConfig(level=logging.INFO, format="(%(threadName)-9s) %(message)s",)


class AsyncBusy(RuntimeError):
    pass


class AsyncNotBusy(RuntimeError):
    pass 


class Processor(ABC):
    def __repr__(self):
        return self.__class__.__name__

    def __str__(self):
        return self.__repr__()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    @abstractmethod
    def process(self, num_frames, *args, **kwargs):
        pass

    @abstractmethod
    def close(self):
        pass


class CameraStreamProcessor(Processor):
    """Camera stream processor.
    """

    def __init__(self, camera, pipeline=[], view=None, display=None, writer=None):
        self.camera = camera
        self.pipeline = pipeline
        self.view = view
        self.display = display
        self.writer = writer

    def process(self, num_frames, outfile=None, start_frame=0):
        # initialization
        if len(self.pipeline) > 0:
            pipeline_func = compose(*self.pipeline[::-1])

        if self.display:
            if len(self.pipeline) > 0:
                display_func = compose(self.display.write, self.view.draw)
            else:
                display_func = self.display.write
            self.display.open()

        if self.writer and outfile:
            self.writer.filename = outfile
            self.writer.open()

        # run pipeline
        results = []
        self._cancel = False
        for i in range(start_frame + num_frames):
            inp = self.camera.read()

            if self._cancel:
                break
            if i < start_frame:
                continue

            if len(self.pipeline) > 0:
                out = pipeline_func(inp)
            else:
                out = inp

            if self.display:
                if len(self.pipeline) > 0:
                    display_func(inp, out)
                else:
                    display_func(inp)

            if self.writer and outfile:
                self.writer.write(inp)

            results.append(out)

        # termination
        if self.display:
            self.display.close()
        if self.writer and outfile:
            self.writer.close()

        return np.array(results)

    def cancel(self):
        self._cancel = True

    def close(self):
        self.camera.close()


class FileStreamProcessor(Processor):
    """File stream processor.
    """
    def __init__(self, reader, pipeline=[], view=None, display=None):
        self.reader = reader
        self.pipeline = pipeline
        self.view = view
        self.display = display

    def process(self, num_frames, infile, start_frame=0):
        # initialization
        self.reader.filename = infile
        self.reader.open()

        if len(self.pipeline) > 0:
            pipeline_func = compose(*self.pipeline[::-1])

        if self.display:
            if len(self.pipeline) > 0:
                display_func = compose(self.display.write, self.view.draw)
            else:
                display_func = self.display.write
            self.display.open()

        # run pipeline
        results = []
        for i, inp in enumerate(self.reader):
            if i < start_frame:
                continue
            if i >= start_frame + num_frames:
                break

            if len(self.pipeline) > 0:
                out = pipeline_func(inp)
            else:
                out = inp

            if self.display:
                if len(self.pipeline) > 0:
                    display_func(inp, out)
                else:
                    display_func(inp)

            results.append(out)

        # termination
        self.reader.close()
        if self.display:
            self.display.close()

        return np.array(results)

    def close(self):
        pass


class CameraStreamProcessorMT(Processor):
    """Camera stream processor.
    """
    def __init__(self, camera, pipeline=[], view=None, display=None, writer=None, qsize=1):
        self.camera=camera
        self.pipeline = pipeline
        self.view = view
        self.display = display
        self.writer = writer
        self.qsize = qsize
        
        self.pipeline_f = None
        self.display_f = None

        if len(self.pipeline) > 0:
            self.camera_out_q = [DataflowQueueMT(self.qsize)]
            self.pipeline_in_q = [self.camera_out_q[0]]
            self.pipeline_out_q = [DataflowQueueMT()]
        else:
            self.camera_out_q = [DataflowQueueMT()]
            
        if self.display:
            self.camera_out_q.append(DataflowQueueMT(self.qsize))
            self.display_in_q = [self.camera_out_q[-1]]
            
            if len(self.pipeline) > 0 and self.view:
                display_func = compose(self.display.write, self.view.draw)
                self.pipeline_out_q.append(DataflowQueueMT(self.qsize))
                self.display_in_q.append(self.pipeline_out_q[-1])
            else:
                display_func = self.display.write

            self.display_f = DataflowFunctionMT(func=display_func,
                                            pre_func=self.display.open,
                                            post_func=self.display.close,
                                            in_queues=self.display_in_q,
                                            )
            self.display_f.start()
          
        if len(self.pipeline) > 0:
            pipeline_func = compose(*self.pipeline[::-1])
            self.pipeline_f = DataflowFunctionMT(func=pipeline_func,
                                             in_queues=self.pipeline_in_q,
                                             out_queues=self.pipeline_out_q)
            self.pipeline_f.start()

    def process(self, num_frames, outfile=None, start_frame=0):
        # Connect writer thread to camera and start running, if required
        writer_f = None
        if outfile and self.writer:
            self.writer.filename = outfile
            self.camera_out_q.append(DataflowQueueMT(self.qsize))
            self.writer_in_q = [self.camera_out_q[-1]]
            writer_f = DataflowFunctionMT(func=self.writer.write,
                                      pre_func=self.writer.open,
                                      post_func=self.writer.close,
                                      in_queues=self.writer_in_q,
                                      )                             
            writer_f.start()

        # Run pipeline
        self._cancel = False
        for i in range(start_frame + num_frames):
            frame = self.camera.read()

            if self._cancel:
                break
            if i < start_frame:
                continue

            for q in self.camera_out_q:
                q.put(frame)
        
        # Flush pipeline
        if len(self.pipeline) > 0:
            for q in self.camera_out_q:
                q.join()
        else:
            for q in self.camera_out_q[1:]:
                q.join()

        # Stop writer thread and disconnect from pipeline
        if writer_f:
            writer_f.in_queues[0].close()
            writer_f.join()
            self.camera_out_q.pop()

        # Get results from pipeline (or camera if no pipeline specified)
        results = []
        if len(self.pipeline) > 0:
            while self.pipeline_out_q[0].qsize() > 0:
                results.append(self.pipeline_out_q[0].get())
        else:
            while self.camera_out_q[0].qsize() > 0:
                results.append(self.camera_out_q[0].get())

        return np.array(results)

    def cancel(self):
        self._cancel = True

    def close(self):           
        self.camera.close()

        if self.display_f:
            self.display_f.in_queues[0].close()
            self.display_f.join()
            
        if self.pipeline_f:
            self.pipeline_f.in_queues[0].close()
            self.pipeline_f.join()

   
class FileStreamProcessorMT(Processor):
    """File stream processor.
    """
    def __init__(self, reader, pipeline=[], view=None, display=None, qsize=1):
        self.reader = reader
        self.pipeline = pipeline
        self.view = view
        self.display = display
        self.qsize = qsize

        self.pipeline_f = None
        self.display_f = None

        if len(self.pipeline) > 0:
            self.reader_out_q = [DataflowQueueMT(self.qsize)]
            self.pipeline_in_q = [self.reader_out_q[0]]
            self.pipeline_out_q = [DataflowQueueMT()]
        else:
            self.reader_out_q = [DataflowQueueMT()]
            
        if self.display:
            self.reader_out_q.append(DataflowQueueMT(self.qsize))
            self.display_in_q = [self.reader_out_q[-1]]
            
            if len(self.pipeline) > 0 and self.view:
                display_func = compose(self.display.write, self.view.draw)
                self.pipeline_out_q.append(DataflowQueueMT(self.qsize))
                self.display_in_q.append(self.pipeline_out_q[-1])
            else:
                display_func = self.display.write

            self.display_f = DataflowFunctionMT(func=display_func,
                                            pre_func=self.display.open,
                                            post_func=self.display.close,
                                            in_queues=self.display_in_q,
                                            )
            self.display_f.start()
                
        if len(self.pipeline) > 0:
            pipeline_func = compose(*self.pipeline[::-1])
            self.pipeline_f = DataflowFunctionMT(func=pipeline_func,
                                             in_queues=self.pipeline_in_q,
                                             out_queues=self.pipeline_out_q)
            self.pipeline_f.start()
        
    def process(self, num_frames, infile, start_frame=0):
        # Get frames from reader and send to pipeline
        self.reader.filename = infile
        self.reader.open()         
        for i, frame in enumerate(self.reader):
            if i < start_frame:
                continue
            if i >= start_frame + num_frames:
                break

            for q in self.reader_out_q:
                q.put(frame)
        self.reader.close()
        
        # Flush pipeline
        if len(self.pipeline) > 0:
            for q in self.reader_out_q:
                q.join()
        else:
            for q in self.reader_out_q[1:]:
                q.join()
        
        # Get results from pipeline (or reader if no pipeline specified)
        results = []
        if len(self.pipeline) > 0:
            while self.pipeline_out_q[0].qsize() > 0:
                results.append(self.pipeline_out_q[0].get())
        else:
            while self.reader_out_q[0].qsize() > 0:
                results.append(self.reader_out_q[0].get())

        return np.array(results)

    def close(self):
        if self.display_f:
            self.display_f.in_queues[0].close()
            self.display_f.join()
            
        if self.pipeline_f:
            self.pipeline_f.in_queues[0].close()
            self.pipeline_f.join()


class CameraStreamProcessorMP(Processor):
    """Camera stream processor.
    """
    def __init__(self, camera, pipeline=[], view=None, display=None, writer=None, qsize=1):
        self.camera = camera
        self.pipeline = pipeline
        self.view = view
        self.display = display
        self.writer = writer
        self.qsize = qsize
        
        self.pipeline_f = None
        self.display_f = None

        if len(self.pipeline) > 0:
            self.camera_out_q = [DataflowQueueMP(self.qsize)]
            self.pipeline_in_q = [self.camera_out_q[0]]
            self.pipeline_out_q = [DataflowQueueMP()]
        else:
            self.camera_out_q = [DataflowQueueMP()]
            
        if self.display:
            self.camera_out_q.append(DataflowQueueMP(self.qsize))
            self.display_in_q = [self.camera_out_q[-1]]
            
            if len(self.pipeline) > 0 and self.view:
                display_func = compose(self.display.write, self.view.draw)
                self.pipeline_out_q.append(DataflowQueueMP(self.qsize))
                self.display_in_q.append(self.pipeline_out_q[-1])
            else:
                display_func = self.display.write

            self.display_f = DataflowFunctionMP(func=display_func,
                                             pre_func=self.display.open,
                                             post_func=self.display.close,
                                             in_queues=self.display_in_q,
                                             )
            self.display_f.start()
          
        if len(self.pipeline) > 0:
            pipeline_func = compose(*self.pipeline[::-1])
            self.pipeline_f = DataflowFunctionMP(func=pipeline_func,
                                              in_queues=self.pipeline_in_q,
                                              out_queues=self.pipeline_out_q)
            self.pipeline_f.start()

    def process(self, num_frames, outfile=None, start_frame=0):
        # Connect writer thread to camera and start running, if required
        writer_f = None
        if outfile and self.writer:
            self.writer.filename = outfile
            self.camera_out_q.append(DataflowQueueMP(self.qsize))
            self.writer_in_q = [self.camera_out_q[-1]]
            writer_f = DataflowFunctionMP(func=self.writer.write,
                                       pre_func=self.writer.open,
                                       post_func=self.writer.close,
                                       in_queues=self.writer_in_q,
                                       )                             
            writer_f.start()

        # Run pipeline
        self._cancel = False
        for i in range(start_frame + num_frames):
            frame = self.camera.read()

            if self._cancel:
                break
            if i < start_frame:
                continue

            for q in self.camera_out_q:
                q.put(frame)
        
        # Flush pipeline
        if len(self.pipeline) > 0:
            for q in self.camera_out_q:
                q.join()
        else:
            for q in self.camera_out_q[1:]:
                q.join()

        # Stop writer thread and disconnect from pipeline
        if writer_f:
            writer_f.in_queues[0].close()
            writer_f.join()
            self.camera_out_q.pop()

        # Get results from pipeline (or reader if no pipeline specified)
        results = []
        if len(self.pipeline) > 0:
            while self.pipeline_out_q[0].qsize() > 0:
                results.append(self.pipeline_out_q[0].get())
        else:
            while self.camera_out_q[0].qsize() > 0:
                results.append(self.camera_out_q[0].get())

        return np.array(results)

    def cancel(self):
        self._cancel = True

    def close(self):           
        self.camera.close()

        if self.display_f:
            self.display_f.in_queues[0].close()
            self.display_f.join()
            
        if self.pipeline_f:
            self.pipeline_f.in_queues[0].close()
            self.pipeline_f.join()


class FileStreamProcessorMP(Processor):
    """File stream processor.
    """
    def __init__(self, reader, pipeline=[], view=None, display=None, qsize=1):
        self.reader = reader
        self.pipeline = pipeline
        self.view = view
        self.display = display
        self.qsize = qsize

        self.pipeline_f = None
        self.display_f = None
        
        self.reader_out_q = [DataflowQueueMP()]
        
        if len(self.pipeline) > 0:
            self.reader_out_q = [DataflowQueueMP(self.qsize)]
            self.pipeline_in_q = [self.reader_out_q[0]]
            self.pipeline_out_q = [DataflowQueueMP()]
        else:
            self.reader_out_q = [DataflowQueueMP()]
            
        if self.display:
            self.reader_out_q.append(DataflowQueueMP(self.qsize))
            self.display_in_q = [self.reader_out_q[-1]]
            
            if len(self.pipeline) > 0 and self.view:
                display_func = compose(self.display.write, self.view.draw)
                self.pipeline_out_q.append(DataflowQueueMP(self.qsize))
                self.display_in_q.append(self.pipeline_out_q[-1])
            else:
                display_func = self.display.write

            self.display_f = DataflowFunctionMP(func=display_func,
                                             pre_func=self.display.open,
                                             post_func=self.display.close,
                                             in_queues=self.display_in_q,
                                             )
            self.display_f.start()
                
        if len(self.pipeline) > 0:
            pipeline_func = compose(*self.pipeline[::-1])
            self.pipeline_f = DataflowFunctionMP(func=pipeline_func,
                                              in_queues=self.pipeline_in_q,
                                              out_queues=self.pipeline_out_q)
            self.pipeline_f.start()
        
    def process(self, num_frames, infile, start_frame=0):
        # Get frames from reader and send to pipeline
        self.reader.filename = infile
        self.reader.open()         
        for i, frame in enumerate(self.reader):
            if i < start_frame:
                continue
            if i >= start_frame + num_frames:
                break

            for q in self.reader_out_q:
                q.put(frame)
        self.reader.close()
        
        # Flush pipeline
        if len(self.pipeline) > 0:
            for q in self.reader_out_q:
                q.join()
        else:
            for q in self.reader_out_q[1:]:
                q.join()
        
        # Get results from pipeline (or reader if no pipeline specified)
        results = []
        if len(self.pipeline) > 0:
            while self.pipeline_out_q[0].qsize() > 0:
                results.append(self.pipeline_out_q[0].get())
        else:
            while self.reader_out_q[0].qsize() > 0:
                results.append(self.reader_out_q[0].get())

        return np.array(results)

    def close(self):
        if self.display_f:
            self.display_f.in_queues[0].close()
            self.display_f.join()
            
        if self.pipeline_f:
            self.pipeline_f.in_queues[0].close()
            self.pipeline_f.join()


class AsyncProcessor(Processor):
    """Asynchronous processor.
    """    
    def __init__(self, sync_processor):
        self.sync_processor = sync_processor
        try:
            self._worker = None
            self._results = queue.Queue()
            self._busy = False
        except:
            self.sync_processor.close()
            raise

    def __getattr__(self, name):
        if self._busy:
            raise AsyncBusy        
        return getattr(self.sync_processor, name)
    
    def __setattr__(self, name, value):
        if name in ():
            if self._busy:
                raise AsyncBusy
            setattr(self.sync_processor, name, value)
        else:
            super().__setattr__(name, value)

    def process(self, num_frames, *args, **kwargs):
        return self.sync_processor.process(num_frames, *args, **kwargs)

    def async_process(self, num_frames=sys.maxsize, *args, **kwargs):
        if self._busy:
            raise AsyncBusy  
        self._busy = True
        self._worker = Thread(target=lambda num_frames, args, kwargs, results: \
                              results.put(self.sync_processor.process(num_frames, *args, **kwargs)), \
                              args=(num_frames, args, kwargs, self._results))
        self._worker.start()
    
    def async_result(self):
        if not self._busy:
            raise AsyncNotBusy            
        self._worker.join()
        result = self._results.get()
        self._busy = False
        return result
    
    def async_done(self):
        if not self._busy:
            raise AsyncNotBusy  
        return not self._worker.is_alive()
    
    def async_cancel(self):
        self.sync_processor.cancel()

    def close(self):
        self.sync_processor.close()
