all: carma_pack

carma_pack: force
	python setup.py build_ext --inplace

clean:
	rm -rf build src/*.pyc src/carma_pack.so

force:
	true
