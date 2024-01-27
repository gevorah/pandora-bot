from recorder import record 
from sr import pipe
# from assistant import prompt

filename = 'output.wav'
record(filename)

sr_result = pipe(filename)
sr_text = sr_result['text']
print(sr_text)

# answer = prompt("How do I attack a wifi network?")
# print(answer)
