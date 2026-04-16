# African Validation CSV Checklist

This project uses a simple CSV file built from locally saved images so validation does not depend on external websites.

## What each row should contain
- `id`: a unique example ID
- `image_path`: the local path to the saved image file
- `text`: the paired caption, claim, or generated text
- `label`: `misinformation` or `likely_consistent`
- `country_focus`: optional but helpful
- `language`: optional but helpful

Store images under `data/african_validation_images/` and reference them with relative paths in the CSV.

## Simple workflow

### 1. Collect an image
- [ ] Save the image locally in `data/african_validation_images/`.
- [ ] Use a stable filename such as `img_0001.jpg`.
- [ ] Confirm the image opens locally before adding it to the CSV.

### 2. Add rows to the CSV
- [ ] Give the row a unique `id`.
- [ ] Fill in `image_path`.
- [ ] Paste or write the paired `text`.
- [ ] Choose a `label`.
- [ ] Add `country_focus` and `language` if known.

### 3. Pairing rule
- [ ] Each saved image should usually appear in two rows.
- [ ] One row should use `likely_consistent`.
- [ ] One row should use `misinformation`.

### 4. Final check
- [ ] Make sure each row has an `id`.
- [ ] Make sure each row has a valid `image_path`.
- [ ] Make sure each row has text.
- [ ] Make sure each row has a valid label.
- [ ] Make sure every referenced image file exists locally.

## Label meanings
- `misinformation`: the image-text pair is misleading or false
- `likely_consistent`: the image and text appear to match and do not seem misleading

## Minimum rule
If the image or the text feels too unclear to judge, skip it instead of forcing a label.
