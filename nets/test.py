import sys
sys.path.append("./")
import cv2
import numpy as np
import mxnet as mx
from detect.mx_mtcnn.mtcnn_detector import MtcnnDetector
from preproccessing.dataset_proc import gen_face, gen_boundbox

MTCNN_DETECT = MtcnnDetector(model_folder=None, ctx=mx.cpu(0), num_worker=1, minsize=50, accurate_landmark=True)


def load_branch(params):
    if params.with_gender:
        return load_C3AE2(params)
    else:
        return load_C3AE(params)     

def load_C3AE(params):
    #models = load_model(pretrain_path, custom_objects={"pool2d": pool2d, "ReLU": ReLU,
    #    "BatchNormalization": BatchNormalization, "tf": tf, "focal_loss_fixed": focal_loss([1] * 12),
    #    "white_norm": white_norm})
    
    from C3AE import build_net
    models = build_net(12, using_SE=params.se_net, using_white_norm=params.white_norm)
    models.load_weights(params.model_path)
    return models

def load_C3AE2(params):
    from C3AE_expand import build_net3, model_refresh_without_nan 
    models = build_net3(12, using_SE=params.se_net, using_white_norm=params.white_norm)
    if params.model_path:
        models.load_weights(params.model_path)
        model_refresh_without_nan(models) ## hot fix which occur non-scientice gpu or cpu
    return models

def predict(models, img, save_image=False):
    try:
        bounds, lmarks = gen_face(MTCNN_DETECT, img, only_one=False)
        ret = MTCNN_DETECT.extract_image_chips(img, lmarks, padding=0.4)
    except Exception as ee:
        ret = None
        print(img.shape, ee)
    if not ret:
        print("no face")
        return img, None
    padding = 200
    new_bd_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT)
    bounds, lmarks = bounds, lmarks

    colors = [(0, 0, 255), (0, 0, 0), (255, 0, 0)]
    for pidx, (box, landmarks) in enumerate(zip(bounds, lmarks)):
        trible_box = gen_boundbox(box, landmarks)
        tri_imgs = []
        for bbox in trible_box:
            bbox = bbox + padding
            h_min, w_min = bbox[0]
            h_max, w_max = bbox[1]
            #cv2.imwrite("test.jpg", new_bd_img[w_min:w_max, h_min:h_max, :])
            tri_imgs.append([cv2.resize(new_bd_img[w_min:w_max, h_min:h_max, :], (64, 64))])

        for idx, pbox in enumerate(trible_box):
            pbox = pbox + padding
            h_min, w_min = pbox[0]
            h_max, w_max = pbox[1]
            new_bd_img = cv2.rectangle(new_bd_img, (h_min, w_min), (h_max, w_max), colors[idx], 2)

        result = models.predict(tri_imgs)
        age, gender = None, None
        if result and len(result) == 3:
            age, _, gender = result
            age_label, gender_label = age[-1][-1], "F" if gender[-1][0] > gender[-1][1] else "M"
        elif result and len(result) == 2:
            age, _  = result
            age_label, gender_label = age[-1][-1], "unknown"
        else:
           raise Exception("fatal result: %s"%result)
        cv2.putText(new_bd_img, '%s %s'%(int(age_label), gender_label), (padding + int(bounds[pidx][0]), padding + int(bounds[pidx][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 2, 175), 2)
    if save_image:
        print(result)
        cv2.imwrite("igg.jpg", new_bd_img)
    return new_bd_img, (age_label, gender_label)

def test_img(params):
    img = cv2.imread(params.image)
    models = load_branch(params)
    predict(models, img, True)

def video(params):
    cap = cv2.VideoCapture(0)
    models = load_branch(params)

    while True:
        ret, img = cap.read()
        ret = True
        img = cv2.resize(img, (640, 480))
        (height, width) = img.shape[:-1]

        if not ret:
            continue
        img, _ = predict(models, img) 
        if img is not None:
            cv2.imshow("result", img)
        if cv2.waitKey(3) == 27:
            break

def load_csv(file_path):
    import base64
    result = []
    with open(file_path, "r") as fd:
        lines = fd.readlines()
        for line in lines:
            num, gender, img_str = line.split(",")
            arr = base64.b64decode(img_str)
            content = np.frombuffer(arr, np.uint8)
            cv_image = cv2.imdecode(content, 1)
            result.append((gender, cv_image))
    return result

def load_local_ano(params):
    result = load_csv("fb4df376a1244c2e2b3f9384ef3bace5.csv")
    models = load_branch(params)
    counter, skip = 0, 0
    for idx, (gender, img) in enumerate(result):
        img, labels = predict(models, img, True)
        if labels is None:
            skip += 1
            continue
        age, pgender = labels
        print("!!!,", pgender)
        if (gender == "male" and pgender == "F") or (gender == "female" and pgender == "M"):
            cv2.imwrite("online/%s_%s_%s.jpg"%(idx, gender, pgender), img)
        else:
            counter += 1
        print(gender, pgender)
    print("%s/ %s /%s"%(counter, skip, len(result)))

def init_parse():
    import argparse
    parser = argparse.ArgumentParser(
        description='C3AE retry')

    parser.add_argument(
        '-g', '--with_gender', action="store_true",
        help='the best model to load')

    parser.add_argument(
        '-m', '--model-path', default="./model/c3ae_model_v2_117_5.830443-0.955", type=str,
        help='the best model to load')
    parser.add_argument(
        '-vid', "--video", dest="video", action='store_true',
        help='use cemera')

    parser.add_argument(
        '-i', "--image", dest="image", type=str, default="./assets/timg.jpg",
        help='use cemera')

    parser.add_argument(
        '-se', "--se-net", dest="se_net", action='store_true',
        help='use SE-NET')

    parser.add_argument(
        '-white', '--white-norm', dest="white_norm", action='store_true',
        help='use white norm')

    params = parser.parse_args()
    return params


if __name__ == "__main__":
    params = init_parse()
    #load_local_ano(params)
    if params.video:
        video(params)
    else:
        if not params.image:
            raise Exception("no image!!")
        test_img(params)
