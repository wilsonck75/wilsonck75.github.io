# AGENTS.md

## Cursor Cloud specific instructions

This is a Jekyll 4.4.x static site (personal portfolio/blog). There is no backend, no database, and no external services.

### Quick reference

| Action | Command |
|--------|---------|
| Install deps | `bundle install` |
| Build site | `bundle exec jekyll build` |
| Dev server | `bundle exec jekyll serve --livereload` |
| Dev server URL | `http://127.0.0.1:4000/` |

### Environment notes

- Ruby is installed system-wide via `apt` (3.2.x). The `.ruby-version` file says `3.2.10` but any 3.2.x works.
- Gems are installed to `vendor/bundle` (local path) to avoid permission issues with `/var/lib/gems`. The bundle config is stored in `.bundle/config`.
- Bundler 4.0.4 is required (matches `Gemfile.lock`).
- The Sass deprecation warnings during build are expected and non-blocking (upstream theme uses `@import` and legacy Sass APIs).
- CI validates builds via `.github/workflows/jekyll.yml` — it runs `bundle exec jekyll build`.
- No lint tool is configured beyond the Jekyll build itself. A successful `jekyll build` is the primary correctness check.
- `_site/` is gitignored output; `vendor/` is also gitignored.
