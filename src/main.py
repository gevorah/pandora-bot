from recorder import record 
from sr import pipe
from assistant import prompt

filename = 'output.wav'
record(filename)

sr_result = pipe(filename)['text']
print(sr_result)

answer = prompt(sr_result)
print(answer)
