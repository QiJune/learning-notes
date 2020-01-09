ccmain :
	clang++ -O3 -std=c++11 adam.cc main.cc -I/usr/local/include/eigen3 -o ccmain

gomain :
	clang++ -O3 -std=c++11 cadam.cc adam.cc -fPIC -shared \
		-I/usr/local/include/eigen3 -o libadam.dylib && \
	go build -o gomain main.go

clean :
	rm -f ccmain gomain *.dylib
