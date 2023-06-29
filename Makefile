# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install:
	@pip install --upgrade pip
	@pip unistall -y preprocessing || : # -y flag avoids asking for confirmation
	@pip install -e .
