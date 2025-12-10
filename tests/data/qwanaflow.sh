#!/bin/bash
qwanaflow --vm-threshold 0.1 --angle-tolerance 5 --stitch-angle-tolerance 20 --ncores 4 \
	test_image.png test_output
