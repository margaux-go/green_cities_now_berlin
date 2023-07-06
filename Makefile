# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install:
	@pip install --upgrade pip
	@pip uninstall -y gcnb_pkg || : # -y flag avoids asking for confirmation
	@pip install -e .

run_preprocess:
	python -c 'from gcnb_pkg.interface.main import preprocess; preprocess("./")'
