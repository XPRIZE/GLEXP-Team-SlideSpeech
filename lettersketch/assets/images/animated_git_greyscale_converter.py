"""
Convert directory of animated .gif files (last frame) to greyscale
"""

import os
files = os.listdir(".")
print(files)

from PIL import Image
for f in files:
	if not f.endswith(".gif"):
		continue
	head = f.split(".")[0]
	im = Image.open(f)
	try:
	    while 1:
	        im.seek(im.tell()+1)
	        im.convert("L").save("grey/" + head + "_.gif")
	        # do something to im
	except EOFError:
	    pass # end of sequence