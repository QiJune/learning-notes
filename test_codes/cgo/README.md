## README

Test in Mac. 

If you want to test in Linux, you should modify Makefile and main.go accordingly.

### Install Eigen

```bash
brew install eigen
```

The eigen will be installed at `/usr/local/include/eigen`


### Build and Run

```bash
make clean
make ccmain && make gomain
./ccmain
./gomain
```

### Test Results

About 100w times adam optimization:

- cgo: 1.28759882s
- c++: 1.17579s
