from pyzbar.pyzbar import decode

def scan_barcode(frame):

    barcodes = decode(frame)

    codes = []

    for barcode in barcodes:

        code = barcode.data.decode("utf-8")
        codes.append(code)

    return codes
