import os 
import numpy as np
import geopandas as gpd

import rasterio
from rasterio.mask import mask
from rasterio import features
import heapq
from shapely.geometry import mapping, shape
from shapely.geometry import box
from shapely.ops import unary_union

from matplotlib import pyplot as plt

###
#Loads and preprocess inputs and outputs results
###

def load_inputs(dem_path, zone_path, sources_path, impact_path):
    """
    Charge et renvoie les données brutes (découpage du DEM inclus).
    Retourne : dict avec clefs : dem, transform, profile, gdf_sources, gdf_impacts
    """
    dem, transform, profile, gdf_poly = load_and_clip_dem(dem_path, zone_path)

    # charger sources et maisons ; maintenir CRS pour reprojection si nécessaire
    gdf_sources = gpd.read_file(sources_path)
    gdf_impacts = gpd.read_file(impact_path)

    # projeter vecteurs dans le CRS du raster si nécessaire
    raster_crs = profile.get('crs')
    if raster_crs is not None:
        if gdf_sources.crs != raster_crs:
            gdf_sources = gdf_sources.to_crs(raster_crs)
        if gdf_impacts.crs != raster_crs:
            gdf_impacts = gdf_impacts.to_crs(raster_crs)

    return {
        'dem': dem,
        'transform': transform,
        'profile': profile,
        'gdf_sources': gdf_sources,
        'gdf_impacts': gdf_impacts,
        'gdf_poly'   : gdf_poly
            }


def load_and_clip_dem(dem_path, poly_path):
    """
    Charge le DEM et le découpe selon le polygone d'étude.
    Retourne (dem_array, transform, profile)
    """
    
    with rasterio.open(dem_path) as src:
        # lire polygone d'étude et l'aligner sur le CRS du raster si besoin
        gdf_poly = gpd.read_file(poly_path)
        if gdf_poly.crs != src.crs:
            gdf_poly = gdf_poly.to_crs(src.crs)
        geoms = [mapping(geom) for geom in gdf_poly.geometry]

        # mask + crop
        dem_crop, transform_crop = mask(src, geoms, crop=True)
        dem_crop = dem_crop[0].astype(float)
        profile  = src.profile.copy()

    return dem_crop, transform_crop, profile, gdf_poly


def plot_speed_field(speed,  title="Speed (m/s)", save_png=None):
    """
    Affiche le champ de vitesse. transform n'est utilisé que si on veut
    récupérer extents (optionnel).
    """
    plt.figure(figsize=(8,6))
    plt.imshow(speed, origin='upper')
    plt.colorbar(label='v (m/s)')
    plt.title(title)
    
    if save_png:
        plt.savefig(save_png, dpi=200)
        plt.show()


def save_raster(array, out_path, profile, transform):
    """
    Sauvegarde un raster float32 avec le profile donné.
    profile doit être un dict issu de rasterio avec width/height mis à jour.
    """
    profile2 = profile.copy()
    profile2.update(dtype=rasterio.float32, count=1, compress='lzw', transform=transform,
    width = array.shape[1], height=array.shape[0], nodata=np.nan)

    with rasterio.open(out_path, 'w', **profile2) as dst:
        dst.write(array.astype(np.float32), 1)


def gdf_to_points(gdf, transform):
     # convertir sources en indices pixel
    points = []
    for geom in gdf.geometry:
        x, y = geom.x, geom.y
        col, row = ~transform * (x, y)
        points.append((int(row), int(col)))
    return points


def isochrone_to_shp(iso_map, transform, out_path, profil=None):
    """
    Polygonize a raster (values 0–100) into 5% interval bins (0–5, 5–10, …, 95–100)
    and export polygons labeled by their upper bin value (0, 5, 10, …, 100).

    Parameters
    ----------
    iso_map : array-like
        Input raster (values 0–100).
    transform : affine.Affine
        Raster transform from the source.
    out_path : str
        Output shapefile path (.shp or .gpkg supported).
    profil : dict, optional
        Raster profile to extract CRS.
    """

    # Convert input to array and create a valid mask
    iso_arr = np.asarray(iso_map)
    mask = np.isfinite(iso_arr)

    if not np.any(mask):
        crs = profil.get("crs") if profil and "crs" in profil else None
        gpd.GeoDataFrame(geometry=[], crs=crs).to_file(out_path)
        print(f"Empty shapefile saved to {out_path}")
        return

    # Define 5% bins: 0–5, 5–10, …, 95–100
    bins = np.arange(0, 105, 5)  # [0,5,10,...,100]
    labels = bins[1:]  # [5,10,...,100] used as class labels

    # Digitize raster into bins (1..20), then map to label values
    class_indices = np.digitize(iso_arr, bins, right=True)
    class_values = np.where(class_indices > 0, labels[np.clip(class_indices - 1, 0, len(labels)-1)], 0)

    # Polygonize
    geom_dict = {}
    for geom_geojson, value in features.shapes(
        class_values.astype(np.int16), mask=mask, transform=transform
    ):
        val = int(value)
        geom_dict.setdefault(val, []).append(shape(geom_geojson))

    # Merge polygons for each bin value (0, 5, 10, …, 100)
    records = []
    for val in sorted(geom_dict.keys()):
        merged = unary_union(geom_dict[val])
        lower = val - 5 if val > 0 else 0
        upper = val
        records.append({
            "class": val,
            "percent_min": lower,
            "percent_max": upper,
            "geometry": merged,
        })

    # Get CRS from profile
    crs = profil.get("crs") if profil and "crs" in profil else None

    # Build GeoDataFrame and save
    gdf = gpd.GeoDataFrame(records, crs=crs)
    gdf.to_file(out_path)
    print(f"Isochrone shapefile saved to: {out_path}")
    return gdf


def save_gdf_to_shp(gdf, out_path):
    """
    sauvegarde un GeoDataFrame dans un shapefile.
    """
    gdf.to_file(out_path)
    return print('Gdf saved to', out_path)


###
#Process functions
###

def create_velocity_field(dem_array, profile, mask, sigma=1.0, load_velocity_raster=None):
    """
    Soit charge un raster de vitesse existant, soit calcule la vitesse depuis le DEM.
    Retourne (speed, profile_speed, slope). mask=True => invalid pixels.
    """
    if load_velocity_raster and os.path.exists(load_velocity_raster):
        with rasterio.open(load_velocity_raster) as src:
            speed = src.read(1).astype(float)
            profile_speed = src.profile.copy()
            return speed, profile_speed, None

    #smooth DEM to reduce noise in slope calculation
    #dem_smooth = gaussian_filter(dem_array, sigma=sigma)

    dx    = profile['transform'][0]
    dy    = abs(profile['transform'][4]) if 'transform' in profile else dx

    # compute slope with NaNs handled
    slope = dem_to_slope(dem_array, dx, dy)

    # convert slope to a movement speed (user-defined function); keep non-zero minima
    speed  = slope_to_fall_speed(slope, dx)
    speed = np.clip(speed, 0.001, 1000)

    # apply provided mask (mask True -> invalid) and keep consistent NaNs
    if mask is not None:
        speed = speed.copy()
        slope = slope.copy()
        speed[mask] = np.nan
        slope[mask] = np.nan

    profile_speed = profile.copy()
    profile_speed.update(dtype=rasterio.float32, count=1)

    return speed, profile_speed, slope


def dem_to_slope(dem, x_res, y_res):
    """Compute slope magnitude (rise/run) handling NaNs.
    - dem : 2D array (float) possibly containing NaNs
    - dx, dy : pixel dimensions (m)
    """
    # Shifted arrays
    dem_up    = np.roll(dem, -1, axis=0)
    dem_down  = np.roll(dem, 1, axis=0)
    dem_left  = np.roll(dem, 1, axis=1)
    dem_right = np.roll(dem, -1, axis=1)

    # Mask out invalid shifts (borders)
    dem_up[-1,:] = np.nan
    dem_down[0,:] = np.nan
    dem_left[:,0] = np.nan
    dem_right[:,-1] = np.nan

    # Compute dx: central differences if both neighbors exist, else forward/backward
    dx = np.where(~np.isnan(dem_right) & ~np.isnan(dem_left),
                (dem_right - dem_left) / (2*x_res),
                np.where(~np.isnan(dem_right),
                        (dem_right - dem)/x_res,
                        np.where(~np.isnan(dem_left),
                                    (dem - dem_left)/x_res,
                                    0)))

    # Compute dy
    dy = np.where(~np.isnan(dem_up) & ~np.isnan(dem_down),
                (dem_down - dem_up) / (2*y_res),
                np.where(~np.isnan(dem_down),
                        (dem_down - dem)/y_res,
                        np.where(~np.isnan(dem_up),
                                    (dem - dem_up)/y_res,
                                    0)))

    # Slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Optional: keep NaN where DEM is NaN
    slope[np.isnan(dem)] = np.nan
    
    return slope


def tobler_speed_from_slope(slope):
    """
    Convertit pente -> vitesse de marche (Tobler) en m/s.
    slope : rise/run (unitless)
    Retour : speed en m/s
    """
    v_kmh = 6.0 * np.exp(-3.5 * np.abs(slope + 0.05)) # km/h

    return (v_kmh / 3.6) # m/s


def slope_to_fall_speed(slope, dx, g=9.81):
    """
    vitesse m/s basée sur l'accélération gravitationnelle
    slope : rise/run
    dx : résolution du raster en m
    """
    h = slope * dx  # hauteur sur la cellule
    speed = np.sqrt(2 * g * np.abs(h))
    return speed


def step_apply_fast_marching(speed, gdf_sources, transform):
    """
    Appelle fast_marching_multi et retourne la matrice T et la liste de pixels sources.
    """
    # convertir sources en indices pixel
    sources_pix = []
    for geom in gdf_sources.geometry:
        x, y = geom.x, geom.y
        col, row = ~transform * (x, y)
        sources_pix.append((int(row), int(col)))

    T = fast_marching_multi(speed, sources_pix)
    return T, sources_pix


def fast_marching_multi(speed, sources_pix, mask=None):
    """
    Fast Marching multi-source on a regular grid, ignoring NaN or masked cells.
    
    Parameters
    ----------
    speed : 2D numpy array (m/s), shape (nrows, ncols)
        Cell velocities. NaN cells are ignored.
    sources_pix : list of (row, col) pixel indices for sources
    mask : 2D boolean numpy array, same shape as speed, optional
        True = masked (ignored) cell, False = valid cell
    
    Returns
    -------
    T : 2D numpy array of travel times (s)
    """
    n, m   = speed.shape
    T      = np.full((n, m), np.inf, dtype=float)
    status = np.zeros((n, m), dtype=np.uint8)  # 0 Far, 1 Narrow, 2 Accepted
    heap   = []

    if mask is None:
        mask = np.isnan(speed)
    else:
        mask = mask.astype(bool)
    
    def valid(i, j):
        """Check if a cell is inside bounds, not accepted, not masked, and has a valid speed"""
        return (0 <= i < n and 0 <= j < m and
                status[i, j] != 2 and
                not mask[i, j] and
                np.isfinite(speed[i, j]) and speed[i, j] > 0)

    def add(i, j):
        """Add a neighbor to the heap if valid"""
        if valid(i, j):
            heapq.heappush(heap, (T[i, j], i, j))
            status[i, j] = 1

    # Initialize sources
    for r, c in sources_pix:
        if valid(r, c):
            T[r, c] = 0.0
            status[r, c] = 2
            # Initial neighbors
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                rr, cc = r+dr, c+dc
                # Check validity before updating
                if valid(rr, cc):
                    T[rr, cc] = min(T[rr, cc], 1.0 / speed[rr, cc])
                    add(rr, cc)

    # Propagation
    while heap:
        t, i, j = heapq.heappop(heap)
        if status[i, j] == 2:
            continue
        status[i, j] = 2
        # Explore neighbors
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            ii, jj = i+dr, j+dc
            # Check validity before updating
            if valid(ii, jj):
                newt = solve_eikonal(T, speed, ii, jj)
                if newt < T[ii, jj]:
                    T[ii, jj] = newt
                add(ii, jj)

    # Set travel time of masked/NaN cells explicitly to np.inf
    T[mask | ~np.isfinite(speed)] = np.inf
    return T



def solve_eikonal(T, speed, i, j):
    """
    Résout localement l'équation d'Eikonal pour la cellule (i,j).
    Retourne la nouvelle estimation du temps minimal.
    """
    n, m = T.shape
    vals = []

    # Récupère les temps voisins valides (finis uniquement)
    vals = []
    for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
        ii, jj = i + di, j + dj
        if 0 <= ii < n and 0 <= jj < m:
            v = T[ii, jj]
            if np.isfinite(v):
                vals.append(v)

    vals = np.sort(vals)
    # Si aucun voisin valide, on ne peut pas calculer
    if len(vals) == 0:
        return np.inf
    
    invs = 1.0 / speed[i, j]

    # Cas avec un seul voisin valide → mise à jour linéaire
    if len(vals) == 1:
        return vals[0] + invs

    # cas général
    a, b = vals[:2]
    
    # Calcul du discriminant en évitant les valeurs invalides
    if not np.isfinite(a) or not np.isfinite(b):
        return np.inf
    
    disc = (a - b) ** 2
    tmp = 2.0 * invs ** 2 - disc
   
    # Si le discriminant est négatif, on fait une mise à jour simple
    if tmp < 0:
        return min(a, b) + invs
    
    # Sinon on fait une mise à jour quadratique
    t = (a + b + np.sqrt(tmp)) / 2.0

    if t >= max(a, b):
        return t
    else:
        return min(a, b) + invs
    

    
def extract_travel_time_for_points(T, transform, gdf_points):
    """
    Extrait les travel times depuis la matrice T pour les points donnés.
    Ajoute une colonne 'travel_time' au GeoDataFrame.
    """
    travel_times = []
    for geom in gdf_points.geometry:
        x, y = geom.x, geom.y
        col, row = ~transform * (x, y)
        row, col = int(round(row)), int(round(col))
        if 0 <= row < T.shape[0] and 0 <= col < T.shape[1]:
            tt = T[row, col]
            if np.isfinite(tt):
                travel_times.append(tt)
            else:
                travel_times.append(np.nan)
        else:
            travel_times.append(np.nan)

    gdf_points = gdf_points.copy()
    gdf_points['travel_time'] = travel_times
    return gdf_points



def classify_by_deciles(gdf_points):
    """
    Classifie les points en déciles selon la colonne 'travel_time'.
    Ajoute une colonne 'decile' au GeoDataFrame.
    """
    gdf_points = gdf_points.copy()
    gdf_points = gdf_points.sort_values('travel_time')

    n       = len(gdf_points)
    deciles = np.ceil((np.arange(1, n+1) / n) * 20).astype(int)*5
    gdf_points['decile'] = deciles
    return gdf_points



def rank_by_travel_time(gdf_points):
    """
    Ajoute une colonne 'rank' au GeoDataFrame selon la colonne 'travel_time'.
    """
    gdf_points = gdf_points.copy()
    gdf_points = gdf_points.sort_values('travel_time')
    gdf_points['rank'] = np.arange(1, len(gdf_points) + 1)
    return gdf_points



def extract_classify_impacts(T, transform, gdf_impacts):
    """
    Étapes finales : extraction travel times, classification, isochrone, sauvegardes.
    - percent : pourcentage d'habitations à retenir pour la capture zone
    """
    # extraction des travel times pour chaque habitation
    gdf_impact_tt = extract_travel_time_for_points(T, transform, gdf_impacts)

    # classification par déciles
    gdf_impact_cl = classify_by_deciles(gdf_impact_tt)
    gdf_impact_rk = rank_by_travel_time(gdf_impact_cl)

    return gdf_impact_rk


def transform_to_isochrone_map(T, transform, profile):
    """
    Transforme la matrice des travel times T en carte d'isochrones classée par pas de 5%.
    - Retourne : (iso_map, profile_iso, class_edges)
      iso_map : uint8 array avec 0 = nodata, 1..20 = classes 5%-bins (1 = 0-5%, 20 = 95-100%)
      profile_iso : profile adapté pour sauvegarde (dtype uint8, nodata=0)
      class_edges : tableau des bords de classes (21 valeurs percentiles 0..100)
    """
    # remplacer les inf par NaN et construire le masque des valeurs valides
    mask = np.isfinite(T)
    T = T.copy()
    T[~mask] = np.nan
    vals = T[mask]

    # cas où il n'y a aucune valeur valide
    if vals.size == 0:
        iso = np.zeros_like(T, dtype=np.uint8)
        profile_iso = profile.copy()
        profile_iso.update(dtype='uint8', count=1, nodata=0)
        class_edges = np.linspace(0, 100, 21)
        return iso, profile_iso, class_edges

    # calcul des bords de classes sur les percentiles 0,5,10,...,100
    class_edges = np.percentile(vals, np.linspace(0, 100, 21))

    # si toutes les valeurs sont identiques, attribuer la classe 5% (valeur 5)
    if np.allclose(class_edges[0], class_edges[-1]):
        iso = np.zeros_like(T, dtype=np.float64)
        iso[mask] = 5
        profile_iso = profile.copy()
        profile_iso.update(dtype='uint8', count=1, nodata=0)
        return iso, profile_iso, class_edges

    # attribution de rangs aux valeurs (0..n-1) pour calculer percentiles de façon robuste
    n = vals.size
    order = np.argsort(vals, kind='mergesort')
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(n, dtype=float)

    # percentiles de chaque valeur (0..100). pour n==1 on évite division par zéro
    if n > 1:
        percentiles = ranks / (n - 1) * 100.0
    else:
        percentiles = np.zeros(n)

    # convertir en classes 5,10,...,100 (par tranches de 5%)
    class_idx = (percentiles // 5).astype(int) + 1
    class_idx = np.clip(class_idx, 1, 20).astype(np.uint8)
    class_values = (class_idx * 5).astype(np.uint8)  # 5..100

    # construire la carte d'isochrones (0 = nodata, 5..100 = pourcentages)
    iso = np.zeros_like(T, dtype=np.float64)
    iso_vals = np.zeros_like(vals, dtype=np.float64)
    iso_vals[:] = class_values
    iso_flat = iso.ravel()
    mask_flat = mask.ravel()
    iso_flat[mask_flat] = iso_vals
    iso = iso_flat.reshape(iso.shape)

    # profil pour écriture raster
    profile_iso = profile.copy()
    profile_iso.update(dtype='float64', count=1, nodata=0)

    return iso, profile_iso, class_edges


def impacts_count(gdf_impacts, transform, profile, dem, resolution, output_path):
    '''
    Compte le nombre d'impacts par cellule de la grille à la résolution demandée
    et calcule la probabilité d'impact par cellule.
    Sauvegarde le résultat dans un shapefile.
    '''
    # adapt DEM to requested resolution (aggregate or upsample)
    orig_res_x = abs(transform.a)
    orig_res_y = abs(transform.e)
    scale_x = resolution / orig_res_x
    scale_y = resolution / orig_res_y

    # no change needed
    if np.isclose(scale_x, 1.0) and np.isclose(scale_y, 1.0):
        pass

    # downsample / aggregate to coarser resolution
    elif scale_x >= 1.0 and scale_y >= 1.0:
        fx = int(round(scale_x))
        fy = int(round(scale_y))
        h, w = dem.shape
        pad_h = int(np.ceil(h / fy) * fy)
        pad_w = int(np.ceil(w / fx) * fx)
        pad_bottom = pad_h - h
        pad_right = pad_w - w
        dem_padded = np.pad(dem, ((0, pad_bottom), (0, pad_right)), constant_values=np.nan)
        dem_reshaped = dem_padded.reshape(pad_h // fy, fy, pad_w // fx, fx)
        dem = np.nanmean(dem_reshaped, axis=(1, 3))

        # update geotransform and profile
        transform = type(transform)(transform.a * fx, transform.b, transform.c,
                                    transform.d, transform.e * fy, transform.f)
        profile = profile.copy()
        profile.update({"width": dem.shape[1], "height": dem.shape[0], "transform": transform})

    # upsample (simple nearest / repeat)
    elif scale_x < 1.0 and scale_y < 1.0:
        ux = int(round(1.0 / scale_x))
        uy = int(round(1.0 / scale_y))
        dem = np.repeat(np.repeat(dem, uy, axis=0), ux, axis=1)

        transform = type(transform)(transform.a / ux, transform.b, transform.c,
                                    transform.d, transform.e / uy, transform.f)
        profile = profile.copy()
        profile.update({"width": dem.shape[1], "height": dem.shape[0], "transform": transform})

    # mixed or non-integer scales: fall back to axis-wise handling (aggregate where >1, repeat where <1)
    else:
        # handle x axis
        if scale_x >= 1.0:
            fx = int(round(scale_x))
        else:
            fx = 1
        # handle y axis
        if scale_y >= 1.0:
            fy = int(round(scale_y))
        else:
            fy = 1

        # if aggregation needed on any axis
        if fx > 1 or fy > 1:
            h, w = dem.shape
            pad_h = int(np.ceil(h / fy) * fy)
            pad_w = int(np.ceil(w / fx) * fx)
            dem_padded = np.pad(dem, ((0, pad_h - h), (0, pad_w - w)), constant_values=np.nan)
            dem_reshaped = dem_padded.reshape(pad_h // fy, fy, pad_w // fx, fx)
            dem = np.nanmean(dem_reshaped, axis=(1, 3))

            transform = type(transform)(transform.a * fx, transform.b, transform.c,
                                        transform.d, transform.e * fy, transform.f)
            profile = profile.copy()
            profile.update({"width": dem.shape[1], "height": dem.shape[0], "transform": transform})

        # if upsampling needed on any axis (after possible aggregation)
        if scale_x < 1.0 or scale_y < 1.0:
            ux = int(round(1.0 / scale_x)) if scale_x < 1.0 else 1
            uy = int(round(1.0 / scale_y)) if scale_y < 1.0 else 1
            dem = np.repeat(np.repeat(dem, uy, axis=0), ux, axis=1)

            transform = type(transform)(transform.a / ux, transform.b, transform.c,
                                        transform.d, transform.e / uy, transform.f)
            profile = profile.copy()
            profile.update({"width": dem.shape[1], "height": dem.shape[0], "transform": transform})
            
    width     = dem.shape[1]
    height    = dem.shape[0]

    polygons = []
    for j in range(height):
        for i in range(width):
            x_min, y_min = transform * (i, j)
            x_max, y_max = transform * (i + 1, j + 1)
            polygons.append(box(x_min, y_min, x_max, y_max))

    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=profile.get('crs'))
    grid['count'] = 0

    # === Comptage des points par cellule ===
    # Jointure spatiale
    join   = gpd.sjoin(gdf_impacts, grid, how="left", predicate="within")
    counts = join.groupby('index_right').size()
    grid.loc[counts.index, 'count'] = counts.values

    # === Calcul de la probabilité ===
    total_points = len(gdf_impacts)
    grid['probability'] = grid['count'] / total_points

    # === Export du shapefile ===
    grid.to_file(output_path)

    print("Fini :")
    print(output_path)
    return grid