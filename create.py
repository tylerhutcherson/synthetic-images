import os
import json
import argparse
import numpy as np
from PIL import Image


parser = argparse.ArgumentParser(description='Create synthetic training data for object detection algorithms.')
parser.add_argument("-bkg", "--backgrounds", type=str, default="Backgrounds/",
                    help="Path to background images folder.")
parser.add_argument("-obj", "--objects", type=str, default="Objects/",
                    help="Path to object images folder.")
parser.add_argument("-o", "--output", type=str, default="TrainingImages/",
                    help="Path to output images folder.")
parser.add_argument("-ann", "--annotate", type=bool, default=True,
                    help="Include annotations in the data augmentation steps?")
parser.add_argument("-s", "--sframe", type=bool, default=True,
                    help="Convert dataset to an sframe?")
parser.add_argument("-g", "--groups", type=bool, default=True,
                    help="Include groups of objects in training set?")
args = parser.parse_args()


# Prepare data creation pipeline
base_bkgs_path = args.backgrounds
bkg_images = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")]
objects_path = args.objects
object_images = [f for f in os.listdir(objects_path) if not f.startswith(".")]
sizes = [0.4, 0.6, 0.8, 1, 1.2] # different obj sizes to use TODO make configurable
count_per_size = 4 # number of locations for each obj size TODO make configurable
annotations = [] # store annots here
output_images = args.output
n = 1


# Helper functions
def get_obj_positions(obj, bkg, count=1):
    obj_w, obj_h = [], []
    x_positions, y_positions = [], []
    bkg_w, bkg_h = bkg.size
    # Rescale our obj to have a couple different sizes
    obj_sizes = [tuple([int(s*x) for x in obj.size]) for s in sizes]
    for w, h in obj_sizes:
        obj_w.extend([w]*count)
        obj_h.extend([h]*count)
        max_x, max_y = bkg_w-w, bkg_h-h
        x_positions.extend(list(np.random.randint(0, max_x, count)))
        y_positions.extend(list(np.random.randint(0, max_y, count)))
    return obj_h, obj_w, x_positions, y_positions


def get_box(obj_w, obj_h, max_x, max_y):
    x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
    x2, y2 = x1 + obj_w, y1 + obj_h
    return [x1[0], y1[0], x2[0], y2[0]]


# check if two boxes intersect
def intersects(box, new_box):
    box_x1, box_y1, box_x2, box_y2 = box
    x1, y1, x2, y2 = new_box
    return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def get_group_obj_positions(obj_group, bkg):
    bkg_w, bkg_h = bkg.size
    boxes = []
    objs = [Image.open(objs_path + obj_images[i]) for i in obj_group]
    obj_sizes = [tuple([int(0.6*x) for x in i.size]) for i in objs]
    for w, h in obj_sizes:
        # set background image boundaries
        max_x, max_y = bkg_w-w, bkg_h-h
        # get new box coordinates for the obj on the bkg
        while True:
            new_box = get_box(w, h, max_x, max_y)
            for box in boxes:
                res = intersects(box, new_box)
                if res:
                    break

            else:
                break  # only executed if the inner loop did NOT break
            #print("retrying a new obj box")
            continue  # only executed if the inner loop DID break
        # append our new box
        boxes.append(new_box)
    return obj_sizes, boxes



if __name__ == "__main__":

    # Make synthetic training data
    print("Making synthetic images.", flush=True)
    for bkg in bkg_images:
        # Load the background image
        bkg_path = base_bkgs_path + bkg
        bkg_img = Image.open(bkg_path)
        bkg_x, bkg_y = bkg_img.size

        # Do single objs first
        for i in obj_images:
            # Load the single obj
            i_path = objs_path + i
            obj_img = Image.open(i_path)

            # Get an array of random obj positions (from top-left corner)
            obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, count=count_per_size)

            # Create synthetic images based on positions
            for h, w, x, y in zip(obj_h, obj_w, x_pos, y_pos):
                # Copy background
                bkg_w_obj = bkg_img.copy()
                # Adjust obj size
                new_obj = obj_img.resize(size=(w, h))
                # Paste on the obj
                bkg_w_obj.paste(new_obj, (x, y))
                output_fp = output_images + str(n) + ".png"
                # Save the image
                bkg_w_obj.save(fp=output_fp, format="png")

                if args.annotate:
                    # Make annotation
                    ann = [{'coordinates': {'height': h, 'width': w, 'x': x+(0.5*w), 'y': y+(0.5*h)}, 'label': i.split(".png")[0]}]
                    # Save the annotation data
                    annotations.append({
                        "path": output_fp,
                        "annotations": ann
                    })
                #print(n)
                n += 1

        if args.groups:
            # 24 Groupings of 2-4 objs together on a single background
            groups = [np.random.randint(0, len(obj_images) -1, np.random.randint(2, 5, 1)) for r in range(2*len(obj_images))]
            # For each group of objs
            for group in groups:
                # Get sizes and positions
                ann = []
                obj_sizes, boxes = get_group_obj_positions(group, bkg_img)
                bkg_w_obj = bkg_img.copy()

                # For each obj in the group
                for i, size, box in zip(group, obj_sizes, boxes):
                    # Get the obj
                    obj = Image.open(objs_path + obj_images[i])
                    obj_w, obj_h = size
                    # Resize it as needed
                    new_obj = obj.resize((obj_w, obj_h))
                    x_pos, y_pos = box[:2]
                    if args.annotate:
                        # Add obj annotations
                        annot = {
                                'coordinates': {
                                    'height': obj_h,
                                    'width': obj_w,
                                    'x': int(x_pos+(0.5*obj_w)),
                                    'y': int(y_pos+(0.5*obj_h))
                                },
                                'label': obj_images[i].split(".png")[0]
                            }
                        ann.append(annot)
                    # Paste the obj to the background
                    bkg_w_obj.paste(new_obj, (x_pos, y_pos))

                output_fp = output_images + str(n) + ".png"
                # Save image
                bkg_w_obj.save(fp=output_fp, format="png")
                if args.annotate:
                    # Save annotation data
                    annotations.append({
                        "path": output_fp,
                        "annotations": ann
                    })
                #print(n)
                n += 1

    if args.annotate:
        print("Saving out Annotations", flush=True)
        # Save annotations
        with open("annotations.json", "w") as f:
            f.write(json.dumps(annotations))

    if args.sframe:
        print("Saving out SFrame", flush=True)
        # Write out data to an sframe for turicreate training
        import turicreate as tc
        # Load images and annotations to sframes
        images = tc.load_images(output_images).sort("path")
        annots = tc.SArray(annotations).unpack().sort("path")
        # Join
        images = images.join(annots)
        # Save out sframe
        images[['image', 'path', 'annotations']].save("training_data.sframe")

    print("Done!", flush=True)
