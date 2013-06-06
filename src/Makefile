all: carmcmcLib

carmcmcLib: force
	CFLAGS="-O3" python setup.py build_ext --inplace

clean:
	rm -rf build lib/carmcmc/*.pyc lib/carmcmc/carmcmcLib.so

force:
	true
