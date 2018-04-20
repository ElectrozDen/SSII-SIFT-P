import ImagesManager

def render_header(file):
    file.write('<html> <link rel="stylesheet" type="text/css" href="../css/bootstrap.min.css"><script src="../js/jquery.js"></script><script src="../js/bootstrap.min.js"></script><div class="container">')
    return

def render_image(file, im: ImagesManager, kmeans):
    file.write('<div class="col-md-12"><h1><span class="badge badge-secondary">Class 0</span></h1>')
    for index, lab in enumerate(kmeans.labels_):
        if lab == 0:
            for file in im.files:

    #for img in images:
     #   file.write('<img src="../../output/keypoints_'+img+'.jpg" class="img-fluid col-md-2" alt="Responsive image" >')
    return

def write_image(file, path: str):
    file.write('<img src="'+path+'" class="img-fluid col-md-2" alt="Responsive image" >')
    return

def render_footer(file):
    file.write('</div></html>')
    return