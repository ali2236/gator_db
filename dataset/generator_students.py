import random

# 100
stids = range(40110000, 40110100)
# 64
cids = range(100, 164)
# 10
sids = range(990, 1000)

records = []

for smester in sids:
    for cid in cids:
        for stid in stids:
            record = f"{stid},{smester},{cid},{random.randint(0, 20)}\n"
            records.append(record)

random.shuffle(records)

with open('grades.csv', 'w') as recordsFile:
    recordsFile.write("student_id, semester, course_id, grade\n")
    recordsFile.write("int, int, int, int\n")
    for record in records:
        recordsFile.write(record)

print('done!')