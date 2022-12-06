if augment == True:

    annot_data_rscale = annot_data.copy()
    annot_data_translate = annot_data.copy()
    annot_data_rotate = annot_data.copy()

    print("Init time: ", datetime.datetime.now())
    print("Initial amount of images: ", len(annot_data['image']))

    def rotate(row, angle):
        new_img, new_bboxs = RandomRotate(angle)(
            row['image'], row['np_bboxes'])
        return new_img, new_bboxs

    annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(
        *annot_data_rotate.apply(lambda row: rotate(row, 180.0), axis=1))
    annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,60), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,90), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,120), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,150), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,180), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,210), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,240), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,270), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)
    # annot_data_rotate["image"], annot_data_rotate["bbox"] = zip(*annot_data_rotate.apply(lambda row: rotate(row,310), axis=1))
    # annot_data = annot_data.append(annot_data_rotate, ignore_index=True)

    print("Final rotate time: ", datetime.datetime.now())

    def scale(row, ratio):
        new_img, new_bboxs = RandomScale(ratio, diff=True)(
            row['image'], row['np_bboxes'])
        return new_img, new_bboxs

    annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(
        *annot_data_rscale.apply(lambda row: scale(row, 0.2), axis=1))
    annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: scale(row,0.4), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: scale(row,0.6), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: scale(row,0.8), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)

    print("Final scale time: ", datetime.datetime.now())

    def translate(row, ratio):
        new_img, new_bboxs = RandomTranslate(
            ratio, diff=True)(row['image'], row['np_bboxes'])
        return new_img, new_bboxs

    annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(
        *annot_data_rscale.apply(lambda row: translate(row, 0.2), axis=1))
    annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: translate(row,0.4), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: translate(row,0.6), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)
    # annot_data_rscale["image"], annot_data_rscale["bbox"] = zip(*annot_data_rscale.apply(lambda row: translate(row,0.8), axis=1))
    # annot_data = annot_data.append(annot_data_rscale, ignore_index=True)

    print("Final translate time: ", datetime.datetime.now())

    annot_data.drop(['np_bboxes', 'path'], axis=1, inplace=True)
    # plotted_img = draw_rect(annot_data['image'][len(annot_data['bbox'])-1].copy(), annot_data['bbox'][len(annot_data['bbox'])-1].copy())
    # plt.imshow(plotted_img)
    # plt.show()

    print("Augmented amount of images: ", len(annot_data['image']))
    print("Final time: ", datetime.datetime.now())