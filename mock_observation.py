from homemade_render import homemade_render
from astropy.visualization import make_lupton_rgb

##load your galaxy with pynbody##
s=pynbody.load('/data/database/nihao/nihao_classic/g8.26e11/g8.26e11.01024')
h=s.halos()
# rotate your galaxy (to face on here)
pynbody.analysis.angmom.faceon(h[1])

###
r,g,b=homemade_render(h[1].s,resolution=500,with_dust=True,width='50 kpc')
# adjust stretch to get a better look, see:
# https://docs.astropy.org/en/stable/visualization/rgb.html
rgb_image = make_lupton_rgb(r, g, b,Q=20,stretch=0.1)
plt.imshow(rgb_image, origin='lower')
