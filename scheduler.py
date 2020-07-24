from subprocess import PIPE, Popen
from time import sleep
from threading import Thread, BoundedSemaphore
import json
import signal

sem = BoundedSemaphore()


nq = []

processes = {}

total_mem = 3000000000

def load_config(id):
    with open('configs/'+id+'.json') as file:
        config = json.load(file)
    return config

def start_process(id):
    config = load_config(id)
    processes[id]['obj'] = Popen(config['cmd'], stdout=PIPE, stdin=PIPE, shell=True) #, text=True, bufsize=1
    print(id, ' started.')



def update(info=False):
    #print('Update called.')
    with sem:
        mem_used = 0
        running = []
        for p in processes:
            pr = load_config(p)['mem_req']
            processes[p]['mem'] = pr
            if processes[p]['obj'].poll() == None:
                mem_used += pr
                running.append(p)
        while nq:
            try:
                pr = load_config(nq[-1])['mem_req']
                #print(total_mem, pr)
                if total_mem - mem_used > pr:
                    processes[nq[-1]] = {'mem': pr}
                    start_process(nq[-1])
                    mem_used += pr
                    nq.pop()
                else:
                    break
            except Exception as exc:
                popped = nq.pop()
                if popped in processes:
                    del processes[popped]
                print(exc)
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
    if tokens[0] == 'stop':
        if tokens[1] == 'id':
            if tokens[2] in processes and processes[tokens[2]]['obj'].poll() == None:
                #processes[tokens[2]]['obj'].terminate() #.send_signal(signal.CTRL_C_EVENT)
                Popen("docker stop "+tokens[2], shell=True)
                print(tokens[2], ' terminated.')
            else:
                print(tokens[2], 'could not be stopped.')
    if tokens[0] == 'setbudget':
        try:
            total_mem = int(tokens[1])
        except BaseException as exc:
            print(exc)
    #if tokens[0] == 'print' and tokens[1] in processes:
    #    print(processes[tokens[1]]['obj'].stdout.read())
    update(info=(tokens[0] == 'info'))
