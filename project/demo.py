import image_artist_style

# image_artist_style.image_client("TAI", "images/*.png", "output")
# image_artist_style.image_server("TAI")

image_artist_style.image_predict("images/content/*.png", "images/style/0011.png", "output")
