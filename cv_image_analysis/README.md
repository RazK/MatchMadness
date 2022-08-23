## to run: 
 - python3 deliver.py
-------
## or
 - import deliver and call main function with an image path as a parameter and an optional display paramter to view the image steps
-------
## improving detections:

- if a square in an image was not found:
call bad_match with the image path
it will open a gui for cropping the square that wasnt matched.


- it will show the entire card with a rectangle around the current square that is displayed in a second frame\
press y to save the square\
press q to quit the image\
press any key to skip square


- finally go to the directory 'templates' and change the image name to the currect template class.\
for example:
orange square is '5', add any unique string (image class is taken from the first char)\
you can rename the image to 5_2.jpg, 5Askndal.jpg etc...

