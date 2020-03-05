#!/usr/bin/python3

print()
print("checking for nltk")
try:
    import nltk
except ImportError:
    print("you should install nltk before continuing")
    print("try using: pip3 install nltk")

print("checking for numpy")
try:
    import numpy
except ImportError:
    print("you should install numpy before continuing")
    print("try using: pip3 install numpy")

print("checking for sklearn")
try:
    import sklearn
except:
    print("you should install sklearn before continuing")
    print("try using: pip3 install sklearn")


print("downloading the Enron dataset (this may take a while)")
print("to check on progress, you can cd up one level, then execute <ls -lthr>")
print("Enron dataset should be last item on the list, along with its current size")
print("download will complete at about 423 MB")
# from urllib.request import urlretrieve
# url = "https://www.cs.cmu.edu/~./enron/enron_mail_20150507.tar.gz"
# urlretrieve(url, filename="../enron_mail_20150507.tar.gz")
print("download complete!")


print("unzipping Enron dataset (this may take a while)")
import tarfile
import os
os.chdir("..")
tfile = tarfile.open("enron_mail_20150507.tar.gz", "r:gz")
tfile.extractall(".")

print("you're ready to go!")
