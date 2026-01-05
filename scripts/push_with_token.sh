#!/bin/bash

# Прочитане на токена от файл
TOKEN=$(cat secrets/github_aurora_key.txt)

# Динамично изграждане на HTTPS URL с токен
REPO_URL="https://${TOKEN}@github.com/kenderovemil/AuroraAI.git"

# Промяна на remote URL временно
git remote set-url origin "$REPO_URL"

# Push към GitHub
git push origin main

# Възстановяване на нормалния (чист) URL
git remote set-url origin https://github.com/kenderovemil/AuroraAI.git
