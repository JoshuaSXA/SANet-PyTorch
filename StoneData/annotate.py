import cv2
img = cv2.imread("9.png")


# print img.shape

def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        f=open("9.txt", "a+")
        f.write(xy+'\n')
        f.close()
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        # cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
        #             1.0, (0, 0, 0), thickness=1)
        cv2.imshow("image", img)


cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
cv2.imshow("image", img)

while (True):
    try:
        cv2.waitKey(0)
    except Exception:
        cv2.destroyAllWindows()
        break



cv2.waitKey(0)
cv2.destroyAllWindows()