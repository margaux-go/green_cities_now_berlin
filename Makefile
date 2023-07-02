# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install:
	@pip install --upgrade pip
	@pip uninstall -y gcnb_pkg || : # -y flag avoids asking for confirmation
	@pip install -e .
