#!/usr/bin/env python

import argparse
import json

import ijson
import numpy as np
import pandas as pd
import shapely.ops
import zarr


def parse_objects(parser):
    while True:
        prefix, event, value = next(parser)
        if event == 'end_array':
            break
        if event == 'start_map':
            builder = ijson.common.ObjectBuilder()
            builder.event(event, value)
            for p, e, v in parser:
                builder.event(e, v)
                if p == prefix and e == 'end_map':
                    yield builder.value
                    break


def parse_valuelist(parser):
    while True:
        prefix, event, value = next(parser)
        if event == 'end_array':
            break
        if event == 'start_array':
            sublist = []
            for _, e, v in parser:
                if e == 'end_array':
                    break
                if e in ('number', 'string', 'boolean', 'null'):
                    sublist.append(v)
            yield sublist


def parse_stringlist(parser):
    result = []
    for _, event, value in parser:
        if event == 'end_array':
            break
        if event == 'string':
            result.append(value)
    return result


def mask_to_polygon_boundaries(mask):
    pixel_squares = []
    for r in range(mask.shape[0]):
        for c in range(mask.shape[1]):
            if mask[r, c]:
                pixel_squares.append(shapely.geometry.box(c, r, c + 1, r + 1))
    if not pixel_squares:
        return []
    union_poly = shapely.ops.unary_union(pixel_squares)
    polys = []
    if isinstance(union_poly, shapely.geometry.Polygon):
        polys = [union_poly]
    elif isinstance(union_poly, shapely.geometry.MultiPolygon):
        polys = list(union_poly.geoms)

    boundaries = []
    for poly in polys:
        simplified = poly.simplify(0, preserve_topology=True)
        xy = list(simplified.exterior.coords)
        xy = [(int(x), int(y)) for x, y in xy]
        if ((xy[0][0] == xy[1][0] and xy[0][0] == xy[-2][0]) or
                (xy[0][1] == xy[1][1] and xy[0][1] == xy[-2][1])):
            xy = xy[1:-1] + xy[1:2]
        boundaries.append(xy)

    return boundaries


def object_to_geojson(obj, fptr, first, bboxes):
    if len(obj['fields']['alRegions']) != 1:
        raise Exception('Surprising alRegions')
    region = obj['fields']['alRegions'][0]['fields']
    x = region['x']
    y = region['y']
    minx = min(x)
    miny = min(y)
    maxx = max(x)
    maxy = max(y)
    bboxes.append([minx, miny, maxx, maxy])
    grid = np.zeros((maxy - miny + 1, maxx - minx + 1), dtype=np.uint8)
    for px, py in zip(x, y):
        grid[py - miny, px - minx] = 1
    poly = [[pt[0] + minx, pt[1] + miny] for pt in mask_to_polygon_boundaries(grid)[0]]
    if poly[0] != poly[-1]:
        poly.append(poly[0])
    feature = {'type': 'Feature', 'geometry': {'type': 'Polygon', 'coordinates': [poly]}}
    if fptr:
        if not first:
            fptr.write(',\n')
        fptr.write(json.dumps(feature, separators=(',', ':')))


def parse_json(filename, geojsonname, zarrname, csvname):
    bboxes = []
    rows = []
    geojsonfptr = None
    if geojsonname:
        geojsonfptr = open(geojsonname, 'w')
        geojsonfptr.write('{"type": "FeatureCollection", "features": [\n')
    parser = ijson.parse(open(filename, 'rb'), use_float=True)
    _, event, _ = next(parser)
    if event != 'start_array':
        raise ValueError('Expected JSON array at top level.')
    _, _, root_id = next(parser)
    print(root_id)
    _, event, _ = next(parser)
    if event != 'start_array':
        raise ValueError('Expected JSON array for element.')
    c1 = 0
    for obj in parse_objects(parser):
        object_to_geojson(obj, geojsonfptr, not c1, bboxes)
        c1 += 1
    _, event, _ = next(parser)
    if event != 'start_array':
        raise ValueError('Expected JSON array for element.')
    c2 = 0
    for row in parse_valuelist(parser):
        rows.append(row + bboxes[c2])
        c2 += 1
    if event != 'start_array':
        raise ValueError('Expected JSON array for element.')
    headers = parse_stringlist(parser)
    headers += ['skip'] * (len(rows[0]) - len(headers) - 4)
    headers += ['bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1']
    print(headers)
    if event != 'start_array':
        raise ValueError('Expected JSON array for element.')
    headers2 = parse_stringlist(parser)
    headers2 += ['skip'] * (len(rows[0]) - len(headers2) - 4)
    headers2 += ['bbox_x0', 'bbox_y0', 'bbox_x1', 'bbox_y1']
    print(headers2)
    _, event, _ = next(parser)
    if event != 'end_array':
        raise ValueError('Expected JSON array to end.')
    print(c1, c2, len(headers), len(headers2))
    if geojsonfptr:
        geojsonfptr.write('\n]}')
    df = pd.DataFrame(rows, columns=headers)
    if csvname:
        df.to_csv(csvname, index=False)
    if zarrname:
        store = zarr.ZipStore(zarrname, mode='w')
        data = df.values
        chunk_size = (1000, len(rows[0]))
        z = zarr.open(store=store, mode='w', shape=data.shape, dtype=data.dtype, chunks=chunk_size)
        z[:] = data
        z.attrs.update({'headers': headers, 'long_headers': headers2})
        store.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Parse a json output from ObjectStreamParser into annotations.')
    parser.add_argument('source', help='Source json file.')
    parser.add_argument('--geojson', help='Output geojson file.')
    parser.add_argument('--zarr', help='Output zarr zip file.')
    parser.add_argument('--csv', help='Output csv file.')
    opts = parser.parse_args()
    parse_json(opts.source, opts.geojson, opts.zarr, opts.csv)
