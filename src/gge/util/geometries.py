def extract_geometry_coordinates(geojson):
    """
    Recursively search for 'geometry' in a GeoJSON structure and extract coordinates.

    :param geojson: The GeoJSON structure.
    :return: The coordinates if found, otherwise None.
    """
    if isinstance(geojson, dict):
        # If the current element is a dictionary, check for 'geometry' key
        if 'geometry' in geojson and 'coordinates' in geojson['geometry']:
            return geojson['geometry']['coordinates']
        else:
            # Recursively search in each value of the dictionary
            for key, value in geojson.items():
                result = extract_geometry_coordinates(value)
                if result is not None:
                    return result
    elif isinstance(geojson, list):
        # If the current element is a list, recursively search in each item
        for item in geojson:
            result = extract_geometry_coordinates(item)
            if result is not None:
                return result
    # Return None if no geometry is found
    return None