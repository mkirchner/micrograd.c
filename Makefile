CC=clang
CFLAGS=-Iinclude -O0 -g

%.o: %.c
	$(CC) -c -o $@ $< $(CFLAGS)

main: src/main.o src/value.o
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f src/*.o main
