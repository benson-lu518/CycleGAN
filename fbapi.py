import facebook as fb
from os import listdir

access_token = "EAACC4vUJ74YBAGWLXB062SOo9qwuOtR2izZAbRgvl17RjgFf6Ub6vsTG6BxP4ZAE30eBTf0PWtftSdjOCwOkjqXWHsiZCLJeaKCxKAh1dM5Fwe5RRP5oPWLXLNvB406stt9rbbpuHTKJB845FNUdk3XyvgyGBxQ0DgxOrj7VwimhYM57nMNKIHYwVHXuCVU1fvZAeJPTfRqaNT32fqGV"
asafb = fb.GraphAPI(access_token)
# asafb.put_object("me","feed",message = "This is automated post by facebook jdk ")
asafb.put_photo(open("./prediction/merge_S__78374307.jpg","rb"), message = "This photo is transferred to painting by CycleGAN and automated posted by the system")

