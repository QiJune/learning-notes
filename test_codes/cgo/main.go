package main

// #cgo CFLAGS: -I/usr/local/include/eigen3
// #cgo LDFLAGS: -L . -ladam
// #include "cadam.h"
import "C"
import "fmt"
import "unsafe"
import "math/rand"
import "time"

func main() {
	const numel int = 1000

	var beta1 float32 = 0.1
	var beta2 float32 = 0.2
	var epsilon float32 = 0.01
	var beta1_pow float32 = 0.3
	var beta2_pow float32 = 0.4

	var mom1 [numel]float32
	var mom2 [numel]float32
	var grad [numel]float32
	var param [numel]float32
	var lr float32 = 0.1

	for i := 0; i < numel; i++ {
		mom1[i] = rand.Float32()
		mom2[i] = rand.Float32()
		grad[i] = rand.Float32()
		param[i] = rand.Float32()
	}

	mom1_ptr := (*C.float)(unsafe.Pointer(&mom1[0]))
	mom1_out_ptr := mom1_ptr
	mom2_ptr := (*C.float)(unsafe.Pointer(&mom2[0]))
	mom2_out_ptr := mom2_ptr
	grad_ptr := (*C.float)(unsafe.Pointer(&grad[0]))
	param_ptr := (*C.float)(unsafe.Pointer(&param[0]))
	param_out_ptr := param_ptr

	start := time.Now()
	for i := 0; i < 1000000; i++ {
		C.c_adam(C.float(beta1),
			C.float(beta2),
			C.float(epsilon),
			C.float(beta1_pow),
			C.float(beta2_pow),
			mom1_ptr,
			mom1_out_ptr,
			mom2_ptr,
			mom2_out_ptr,
			C.float(lr),
			grad_ptr,
			param_ptr,
			param_out_ptr,
			C.int(numel))
	}
	elapsed := time.Since(start)
	fmt.Println(elapsed)
}
