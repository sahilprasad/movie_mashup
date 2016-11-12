count = set()
with open('bee.txt', 'r') as f:
    for line in f:
        line = line.strip()
        line = line.split()
        for word in line:
            count.add(word)

print len(count)

