.PHONY: clean-pyc

clean-pyc:
	bash -c 'find . | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf'