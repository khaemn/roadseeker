# import mss
# import mss.tools
#
#
# with mss.mss() as sct:
#     # The screen part to capture
#     monitor = {"top": 160, "left": 160, "width": 160, "height": 135}
#     output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)
#
#     # Grab the data
#     sct_img = sct.grab(monitor)
#
#     # Save to the picture file
#     mss.tools.to_png(sct_img.rgb, sct_img.size, output=output)
#     print(output)
#
# import numpy as np
# import cv2
# from mss import mss
# from PIL import Image
#
# mon = {'top': 160, 'left': 160, 'width': 200, 'height': 200}
#
# sct = mss()
#
# while 1:
#     sct.get_pixels(mon)
#     img = Image.frombytes('RGB', (sct.width, sct.height), sct.image)
#     cv2.imshow('test', np.array(img))
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break


from PIL import Image
import mss as MSS
import numpy as np
import cv2

sct = MSS.mss()

monitor = {"top": 500, "left": 500, "width": 500, "height": 500}
output = "sct-{top}x{left}_{width}x{height}.png".format(**monitor)

while True:
    # Grab the data
    sct_img = sct.grab(monitor)

    img = Image.frombytes('RGB', (sct_img.width, sct_img.height), sct_img.rgb)

    cv2.imshow('test2', np.array(img))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break
