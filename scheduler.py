from subprocess import PIPE, Popen, check_output
from time import sleep, time
from threading import Thread, BoundedSemaphore
import json

sem = BoundedSemaphore()


nq = []

processes = {}

last_start = 0

def get_free_memory():
    result = check_output(
        [
            'nvidia-smi', '--query-gpu=memory.free',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    return int(result)#gpu_memory_map

def load_config(id):
    with open('configs/'+id+'.json') as file:
        config = json.load(file)
    return config

def start_process(id):
    config = load_config(id)
    f = open('outputs/'+id+'.txt', 'w')
    processes[id]['out'] = f
    processes[id]['obj'] = Popen(config['cmd'], stdout=f, shell=True, text=True) #, text=True, bufsize=1
    print(id, ' started.')



def update(info=False):
    #print('Update called.')
    with sem:
        global last_start
        running = {}
        for p in processes:
            try:
                if processes[p]['obj'].poll() == None:
                    if info:
                        running[p] = load_config(p)['progress']
                else:
                    #out, _ = processes[p]['obj'].communicate()
                    #last = '\n'.join(out.split('\n')[-20:])
                    #print(last)
                    processes[p]['out'].close()
                    del processes[p]
                    break
            except BaseException as exc:
                print('Exception!', exc)
        if nq and (time()-last_start > 60):
            try:
                pr = load_config(nq[-1])['mem_req']
                #print(total_mem, pr)
                if get_free_memory() > pr:
                    processes[nq[-1]] = {'mem': pr}
                    start_process(nq[-1])
                    nq.pop()
                    last_start = time()
            except BaseException as exc:
                popped = nq.pop()
                if popped in processes:
                    del processes[popped]
                print('Exception!', exc)
                print(popped, ' removed from queue.')
        if info:
            print('Running: ', running)
            print('Queue: ', nq)
            print(processes)


def automatic_update():
    while True:
        sleep(10)
        update()


update_t = Thread(target=automatic_update)
update_t.start()

while True:
    inp = input()
    tokens = inp.split(' ')
    if tokens[0] == 'add':
        if tokens[1] == 'id':
            nq.insert(0, tokens[2])
            print(tokens[2], ' added to queue.')
        if tokens[1] == 'range':
            for i in range(int(tokens[3]), int(tokens[4])):
                nq.insert(0, tokens[2]+str(i))
            print('Range added to queue.')
    if tokens[0] == 'stop':
        if tokens[1] == 'id':
            if tokens[2] in processes and processes[tokens[2]]['obj'].poll() == None:
                #processes[tokens[2]]['obj'].terminate() #.send_signal(signal.CTRL_C_EVENT)
                Popen("docker stop "+tokens[2], shell=True)
                print(tokens[2], ' terminated.')
            else:
                print(tokens[2], 'could not be stopped.')
    #if tokens[0] == 'print' and tokens[1] in processes:
    #    print(processes[tokens[1]]['obj'].stdout.read())
    update(info=(tokens[0] == 'info'))
