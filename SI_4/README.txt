This directory contains Matlab code for generating movies (gratings,
moving bars, and full-field flashes), as well as ready natural images
and natural movies used in the layer 4 simulations.  In the case of
moving bars and flashes, the pre-generated movies are provided too.
Such movies are not provided for gratings, because of very large size
of the movies for all different grating conditions.

NOTE -- all paths in the scripts are provided according to the original
filesystem.  They should be changed to reflect the location of necessary files
in the actual user's filesystem.


1. GRATINGS.

The files are in the directory movies_gratings/.

The file ORI_data.mat contains information about all grating conditions used
for generation of movies (which is more than has been used in the layer 4
simulations).

The file res_192_metadata.txt contains all the same information in text format
(it is produced when the gratings are generated, for convenience).  the
columns in this file are as follows.  The first column is the file name where
the grating movie is saved, the second is the grating direction of movement in degrees,
the third is the spatial frequency in cycles per degree, the fourth is the
temporal frequency in Hz, and the fifth is the contrast.

NOTE -- here we always use contrast 0.8, though it can be changed in the script.  In the
layer 4 work, we changed the contrast when necessary by updating the function for creating 
LGN firing rates from the movies (see the directory for generating LGN filters and their outputs).

The grating movies are generated using the script create_grating.m.  All
parameters are defined in the script.

NOTE -- a user may adjust the resolution of the movie.  We used 192 x 96
resolution.  For some of the movies with very fine spatial frequency, we used
twice higher resolution.  However, those movies were not used in the layer 4 work.

NOTE -- the output files (mat files containing the grating movies) are written
out to disk.  Make sure the path to the output directory is  consistent with the user's
directory structure.  The current version of the script writes movie files out to the
directory res_192/, which is provided for convenience.  However, the grating movies
for all parameters we sampled combine to a very large disk volume, and, thus, they
are not provided here.


2. FULL-FIELD FLASHES.

The files are in the directory movies_flashes/.

The script create_flashes.m creates two types of white flashes on gray background.
Flash 1 starts at 1,000 ms and ends at 2,000 ms, and flash 2 starts at 600 ms
and ends at 650 ms.  These parameters are specified in the script.  Update the
values of "kk" and conditions for "i" (e,.g., "if i > 600" or "if i > 650") to
switch between flash 1 and flash 2.

The saved movies of these flashes are available as well:
flash_1.mat and flash_2.mat.


3. MOVING BARS.

The files are in the directory movies_bars/.

The script temp_mov_bar.m creates a few types of white or black moving bars on gray
background.  All parameters are defined in the script.  Note that the output
movie name needs to be updated manually, e.g.,

out_name = 'Wbar_v50pixps_vert.mat';

The saved movies of these bars are available as well:

Bbar_v50pixps_hor.mat
Bbar_v50pixps_vert.mat
Wbar_v50pixps_hor.mat
Wbar_v50pixps_vert.mat


4. NATURAL IMAGES.

There is no script for creating movies of natural image presetnation.
Instead, we used the function for generating LGN firing rates in such a way
that it takes in a list of image files and combines them into a
sequence of matrices of appropriate size.  The 10 images used in the layer 4
work are provided in the directory natural_images/.

The image file names are as follows.

img011_BR.tiff
img019_BT.tiff
img024_BT.tiff
img049_BT.tiff
img057_BR.tiff
img062_VH.tiff
img069_VH.tiff
img071_VH.tiff
img090_VH.tiff
img101_VH.tiff


5. NATURAL MOVIES.

Like for the natural images, no script for creating natural movies was
provided.  Instead, we used npy arrays representing portions of the movie of
interest.  The arrays were used as inputs to the function producing the LGN
firing rates for natural movies.  The arrays for the three natural movie clips
we used are in the directory natural_movies/.

The movie file names are as follows.

TouchOfEvil_frames_1530_to_1680.npy
TouchOfEvil_frames_3600_to_3750.npy
TouchOfEvil_frames_5550_to_5700.npy

NOTE -- in the layer 4 work, we did a small number of simulations with the
movie frames shuffled or pixels within each frame shuffled.  In stead of
creating separate movies for those cases, we shuffled the frames or pixels in
the function that was used to create LGN firing rates from the movies (see the
directory for generating LGN filters and their outputs).


