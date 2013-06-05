all: carmcmc

carmcmc: force
	python setup.py build_ext --inplace

clean:
	rm -rf build src/*.pyc src/carmcmc.so

force:
	true
