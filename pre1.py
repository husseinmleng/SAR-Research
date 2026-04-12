import cv2

def preprocess_image(img, crop_size=(256,256), out_size=(224,224)):
    h, w = img.shape[:2]
    ch, cw = crop_size

    ch = min(ch, h)
    cw = min(cw, w)

    y = (h - ch) // 2
    x = (w - cw) // 2

    img = img[y:y+ch, x:x+cw]
    img = cv2.resize(img, out_size, interpolation=cv2.INTER_LINEAR)
    return img