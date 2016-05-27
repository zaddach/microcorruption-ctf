#!/usr/bin/env python

import os
import sys
import argparse
import bs4
import re

RE_NAME = re.compile("([0-9a-f]{4}) <([^>]*)>$")

def main(args, env):
    soup = bs4.BeautifulSoup(sys.stdin, "lxml")
    for tag in soup.find_all('div', class_ = "insnlabel"):
        match = RE_NAME.match(tag.pre.text)
        if match:
            sys.stdout.write("idc.MakeName(0x%04x, \"%s\")\n" % (int(match.group(1), 16), match.group(2)))

def parse_args():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    code = main(parse_args(), os.environ)
    if code:
        sys.exit(code)
	
