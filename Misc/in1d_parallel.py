import numpy
import IPython
from multiprocessing import Process,Queue

def DoWork_in1d(ArrayA,ArrayB,ThisQueue):
    res = numpy.in1d(ArrayA,ArrayB, assume_unique = True)
    ThisQueue.put(res)
    return True

def in1d_parallel(Array1,Array2,Ncpu = 4):
    
    Len = Array1.shape[0]
    NPerCpu = Len/Ncpu
    Rest = Len - Ncpu*NPerCpu
    

    First = numpy.zeros(Ncpu,dtype=numpy.int64)
    Last = First + 0
    
    for i in range(Ncpu):
        First[i] = i*NPerCpu
    
    Last[:-1] = First[1:]+0
    Last[-1] = First[-1]+NPerCpu+Rest
    
    
    
    Processes = []
    Queues = []
    for i in range(Ncpu):
        Queues.append(Queue())
        Processes.append(Process(target=DoWork_in1d,args=(Array1[First[i]:Last[i]],Array2,Queues[i])))
        Processes[i].start()

    
    Res = numpy.array([],dtype=bool)
    for i in range(Ncpu):
        Res = numpy.append(Res, Queues[i].get())
    
    for i in range(Ncpu):
        Processes[i].join()
    
    return Res