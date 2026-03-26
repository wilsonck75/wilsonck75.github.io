# wilsonck75.github.io

Personal Jekyll site for [Charlie Wilson](https://wilsonck75.github.io), built with fully local Jekyll theme files and published via GitHub Pages.

## Stack

- Jekyll 4.4.x
- Ruby 3.2.x
- Local theme files vendored from `analytics-link/analytics-link.github.io`

## Project Structure

- `_posts/`: blog and project posts
- `legacy/`: redirect pages that preserve older published URLs
- `img/`: site images and post assets
- `docs/`: downloadable documents
- `_includes/`: local overrides for key theme partials such as header, footer, and post cards
- `_layouts/`, `_sass/`, `assets/`: fully local theme structure used for rendering the site
- `_config.yml`: site metadata, plugins, and build configuration
- `index.html`, `tags.html`, `404.html`: top-level pages

## Local Development

1. Install Ruby `3.2.x`.
   On macOS with Homebrew: `brew install ruby@3.2`
2. Add Homebrew Ruby to your shell `PATH`.
   Example: `export PATH="/opt/homebrew/opt/ruby@3.2/bin:$PATH"`
3. Install the Bundler version locked by the repo:
   `gem install bundler:4.0.4`
4. Install dependencies:
   `bundle install`
5. Run the local server:
   `bundle exec jekyll serve --livereload`
6. Build the site:
   `bundle exec jekyll build`

The generated site is written to `_site/`, which is intentionally gitignored.

## Content Conventions

- Add posts to `_posts/` using the `YYYY-MM-DD-title.md` naming format.
- Keep images for posts under `img/posts/`.
- Use explicit `permalink` values for public-facing project URLs.
- Keep categories slug-style, such as `data-science` and `machine-learning`.
- `docs/CharlieWilson_CV.pdf` is the resume file currently referenced by the site config.

## Maintenance Notes

- Repo-only files such as `README.md`, `WARP.md`, and GitHub workflow files are excluded from the published site.
- A GitHub Actions workflow runs `bundle exec jekyll build` on pushes and pull requests to catch build regressions early.
- Legacy redirect pages preserve older category-based URLs after permalink changes.
- The site no longer depends on the `jekyll-remote-theme` plugin; theme files now live in this repository.
- The local theme files were originally vendored from `analytics-link/analytics-link.github.io` (MIT) and can now be modified independently here.

