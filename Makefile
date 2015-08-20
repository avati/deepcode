

all:
	$(CC) -std=c99 deepencode.c -o deepencode -lm

test: all
	dd if=/dev/urandom of=smalltest bs=4096 count=1
	./deepencode smalltest
