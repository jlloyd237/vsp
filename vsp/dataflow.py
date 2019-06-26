# -*- coding: utf-8 -*-
"""Dataflow classes for multi-threaded and multi-process dataflow operations.
"""

import logging
import queue
import threading
import multiprocessing as mp
import multiprocessing.queues as mpq
from functools import partial
from time import sleep


logging.basicConfig(level=logging.INFO, format="(%(threadName)-9s) %(message)s",)


class DataflowQueueMT(queue.Queue):
    CLOSE_SENTINEL = object()
    
    def __init__(self, maxsize=0):
        super().__init__(maxsize)
        
    def __iter__(self):
        while True:
            item = self.get()
            if item is self.CLOSE_SENTINEL:
                return
            yield item

    def close(self):
        self.put(self.CLOSE_SENTINEL)
                

class DataflowFunctionMT(threading.Thread):
    def __init__(self, func, pre_func=None, post_func=None,
                 in_queues=[], out_queues=[]):
        super().__init__()
        self.func = func
        self.pre_func = pre_func
        self.post_func = post_func
        self.in_queues = in_queues
        self.out_queues = out_queues
    
    def run(self):
        if self.pre_func:
            self.pre_func()

        for args in zip(*self.in_queues):
            logging.debug("running with args = {}".format(args))           
            args = [a for a in args if a is not None]
            result = self.func(*args) if args else self.func()
            for out_queue in self.out_queues:
                out_queue.put(result)
            for in_queue in self.in_queues:
                in_queue.task_done()
            sleep(0.01)

        if self.post_func:
            self.post_func()


class DataflowQueueMP(mpq.JoinableQueue):
    def __init__(self, maxsize=0):
        ctx = mp.get_context()
        super().__init__(maxsize, ctx=ctx)
        self.is_open = True
        
    def __iter__(self):
        while True:
            item, close_sentinel = super().get()
            if close_sentinel:
                return
            yield item

    def put(self, item):
        super().put((item, False))
        
    def get(self):
        item, close_sentinel = super().get()
        return item

    def close(self):
        if self.is_open:
            super().put((None, True))
        self.is_open = False   
        super().close()


class DataflowFunctionMP(mp.Process):
    def __init__(self, func, pre_func=None, post_func=None, in_queues=[],
                 out_queues=[], obj_class=None, obj_args=(), obj_kwargs={}):
        super().__init__()
        self.func = func
        self.pre_func = pre_func
        self.post_func = post_func
        self.in_queues = in_queues
        self.out_queues = out_queues
        self.obj_class = obj_class
        self.obj_args = obj_args
        self.obj_kwargs = obj_kwargs
        
    def run(self):       
        if self.obj_class is not None:
            obj = self.obj_class(*self.obj_args, **self.obj_kwargs)           
            self.func = partial(self.func, obj)          
            if self.pre_func:
                self.pre_func = partial(self.pre_func, obj)
            if self.post_func:
                self.post_func = partial(self.post_func, obj)

        if self.pre_func:
            self.pre_func()

        for args in zip(*self.in_queues):
            logging.debug("running with args = {}".format(args))           
            args = [a for a in args if a is not None]
            result = self.func(*args) if args else self.func()
            for out_queue in self.out_queues:
                out_queue.put(result)
            for in_queue in self.in_queues:
                in_queue.task_done()
            sleep(0.01)

        if self.post_func:
            self.post_func()
