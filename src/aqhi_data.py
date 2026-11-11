"""Read the supplied AQHI samples without altering raw files.

Scope: simplified-Chinese monthly CSV and RSS, plus the CSDI GeoJSON ZIP.
Missing AQHI is never imputed. Official '*' and '10+' are retained.
"""
from pathlib import Path
from io import StringIO
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
import pandas as pd

# Explicit, reviewable station mapping; do not use fuzzy matching silently.
STATIONS = [
    ('中西区', '中西區', 'Central/Western'), ('南区', '南區', 'Southern'),
    ('东区', '東區', 'Eastern'), ('观塘', '觀塘', 'Kwun Tong'),
    ('深水埗', '深水埗', 'Sham Shui Po'), ('葵涌', '葵涌', 'Kwai Chung'),
    ('荃湾', '荃灣', 'Tsuen Wan'), ('将军澳', '將軍澳', 'Tseung Kwan O'),
    ('元朗', '元朗', 'Yuen Long'), ('屯门', '屯門', 'Tuen Mun'),
    ('东涌', '東涌', 'Tung Chung'), ('大埔', '大埔', 'Tai Po'),
    ('沙田', '沙田', 'Sha Tin'), ('北区', '北區', 'North'),
    ('塔门', '塔門', 'Tap Mun'), ('铜锣湾', '銅鑼灣', 'Causeway Bay'),
    ('中环', '中環', 'Central'), ('旺角', '旺角', 'Mong Kok'),
]
NAME_MAP = {name: en for sc, tc, en in STATIONS for name in (sc, tc, en)}


def station_key(name):
    name = str(name).strip().replace('空氣質素監測站', '')
    if name not in NAME_MAP:
        raise ValueError(f'Unknown station: {name!r}; review mapping before continuing')
    return NAME_MAP[name]


def parse_aqhi(value):
    """Keep capped readings separate: 10+ is not the exact number 11."""
    raw = str(value).strip()
    substituted = '*' in raw
    token = raw.replace('*', '').strip()
    missing = token == ''
    unavailable = token == '-'
    capped = token == '10+'
    valid = missing or unavailable or capped or bool(re.fullmatch(r'(?:[1-9]|10)', token))
    number = int(token) if valid and not missing and not capped and not unavailable else None
    risk = ('Serious' if capped else
            'Low' if number is not None and number <= 3 else
            'Moderate' if number is not None and number <= 6 else
            'High' if number == 7 else
            'Very High' if number is not None else None)
    return dict(aqhi_raw=raw, aqhi_value=number, is_substituted=substituted,
                is_10_plus=capped, is_missing=missing, is_unavailable=unavailable, is_invalid=not valid,
                risk_category=risk)


def read_history(path):
    """Forward-fill date labels only; preserve blank AQHI measurements."""
    text = Path(path).read_text(encoding='utf-8-sig')
    lines = text.splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('日期,时间,'))
    wide = pd.read_csv(StringIO('\n'.join(lines[header:])), dtype=str,
                       keep_default_na=False)
    wide['日期'] = wide['日期'].replace('', pd.NA).ffill()
    # Daily summary rows are not hourly observations. Keep raw source untouched.
    summary_rows = int((wide['时间'] == '每日最高值').sum())
    wide = wide.loc[wide['时间'] != '每日最高值'].copy()
    hours = pd.to_numeric(wide['时间'], errors='raise')
    if not hours.between(1, 24).all():
        raise ValueError('Unexpected hour labels; inspect source convention')
    # Interpret labelled 24:00 as midnight at the start of the following day.
    # Preserve original date/hour for checking this convention against the dictionary.
    wide['timestamp'] = (pd.to_datetime(wide['日期'], format='%Y-%m-%d')
                         + pd.to_timedelta(hours, unit='h')).dt.tz_localize('Asia/Hong_Kong')
    result = wide.melt(id_vars=['日期', '时间', 'timestamp'],
                       var_name='station_source', value_name='value')
    result = result.rename(columns={'日期': 'source_date', '时间': 'source_hour'})
    result['station_id'] = result['station_source'].map(station_key)
    values = pd.DataFrame(result.pop('value').map(parse_aqhi).tolist())
    result = pd.concat([result.reset_index(drop=True), values], axis=1)
    result['source_file'] = Path(path).name
    result.attrs['excluded_daily_summary_rows'] = summary_rows
    return result


def read_network(path):
    with zipfile.ZipFile(path) as archive:
        entries = [n for n in archive.namelist() if n.endswith('.geojson')]
        if len(entries) != 1:
            raise ValueError('Expected exactly one GeoJSON file')
        obj = json.loads(archive.read(entries[0]))
    rows = []
    for feature in obj['features']:
        p = feature['properties']
        lon, lat = feature['geometry']['coordinates'][:2]
        if feature['geometry']['type'] != 'Point' or not (113 < lon < 115 and 22 < lat < 23):
            raise ValueError('Unexpected geometry or coordinates; inspect CRS')
        if abs(lon-float(p['Longitude'])) > 1e-5 or abs(lat-float(p['Latitude'])) > 1e-5:
            raise ValueError('Geometry differs from longitude/latitude attributes')
        rows.append(dict(station_id=station_key(p['NameTC']), name_en=p['NameEN'],
                         name_tc=p['NameTC'], station_type=p['AqmsTypeEN'],
                         longitude=lon, latitude=lat, source_last_update=p.get('LastUpdate')))
    result = pd.DataFrame(rows)
    if result.station_id.duplicated().any():
        raise ValueError('Duplicate station identifiers in network')
    return result


def read_current(path):
    root = ET.parse(path).getroot()
    rows = []
    for item in root.findall('./channel/item'):
        description = item.findtext('description') or ''
        match = re.search(r'[:：]\s*(10\+|10|[1-9])([*]?)\s+(.+?)\s+-\s+', description)
        if not match:
            raise ValueError(f'Unrecognized RSS description: {description}')
        rows.append(dict(station_id=station_key(item.findtext('title')),
                         station_source=item.findtext('title'),
                         published_at=parsedate_to_datetime(item.findtext('pubDate')),
                         official_risk_text=match.group(3), description_raw=description,
                         **parse_aqhi(match.group(1)+match.group(2))))
    if not rows:
        raise ValueError('RSS contains no items')
    return pd.DataFrame(rows)


def audit(history, current, network):
    """Report problems; do not silently remove or fill suspect records."""
    joined = history.merge(network, on='station_id', how='left', validate='many_to_one', indicator=True)
    current_join = current.merge(network, on='station_id', how='left', validate='many_to_one', indicator=True)
    expected = pd.date_range(history.timestamp.min(), history.timestamp.max(), freq='h')
    gaps = {s: len(expected.difference(g.timestamp)) for s, g in history.groupby('station_id')}
    return dict(history_rows=len(history), history_stations=history.station_id.nunique(),
                excluded_daily_summary_rows=history.attrs.get('excluded_daily_summary_rows', 0),
                history_start=str(history.timestamp.min()), history_end=str(history.timestamp.max()),
                missing_values=int(history.is_missing.sum()),
                unavailable_dash_values=int(history.is_unavailable.sum()),
                substituted_values=int(history.is_substituted.sum()),
                capped_values=int(history.is_10_plus.sum()), invalid_values=int(history.is_invalid.sum()),
                duplicate_station_hours=int(history.duplicated(['station_id', 'timestamp']).sum()),
                missing_hours_within_observed_range=gaps,
                network_stations=len(network), current_stations=current.station_id.nunique(),
                current_published_at=sorted(current.published_at.astype(str).unique().tolist()),
                history_unmatched=int((joined['_merge'] != 'both').sum()),
                current_unmatched=int((current_join['_merge'] != 'both').sum()))
