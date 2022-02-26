#!/bin/bash
DIR=$(dirname "$0")
mkdir -p "$DIR"/src/tilesets
wget 'https://nethackwiki.com/mediawiki/images/7/73/3.6.1tiles32.png' -P "$DIR"/src/tilesets
