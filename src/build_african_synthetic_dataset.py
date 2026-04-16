from __future__ import annotations

import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import requests


OUTPUT_PATH = Path("docs/african_validation/template.csv")
IMAGE_DIR = Path("data/african_validation_images")
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "MBDMultimodalMisinformation/1.0 (research project contact: local-user)"
TARGET_IMAGES = 100
THUMB_WIDTH = 320


@dataclass(frozen=True)
class QuerySpec:
    country: str
    topic: str
    likely_template: str
    misinformation_template: str


QUERY_SPECS = [
    QuerySpec("Kenya", "market", "This image shows a market scene in Kenya.", "This image shows panic buying during a national food crisis in Kenya this week."),
    QuerySpec("Kenya", "school", "This image shows a school setting in Kenya.", "This image shows a school shutdown after a violent incident in Kenya this week."),
    QuerySpec("Kenya", "street", "This image shows an everyday street scene in Kenya.", "This image shows unrest on the streets of Kenya today."),
    QuerySpec("Kenya", "wildlife", "This image shows wildlife or conservation activity in Kenya.", "This image shows wildlife dying because of a sudden environmental disaster in Kenya this week."),
    QuerySpec("Kenya", "hospital", "This image shows a health-related setting in Kenya.", "This image shows hospitals collapsing under a deadly outbreak in Kenya."),
    QuerySpec("Nigeria", "market", "This image shows a market scene in Nigeria.", "This image shows severe shortages and emergency rationing in Nigeria this week."),
    QuerySpec("Nigeria", "school", "This image shows a school or education setting in Nigeria.", "This image shows students fleeing a major attack on a school in Nigeria."),
    QuerySpec("Nigeria", "street", "This image shows an everyday street scene in Nigeria.", "This image shows mass unrest in a major Nigerian city today."),
    QuerySpec("Nigeria", "farm", "This image shows farming or agriculture in Nigeria.", "This image shows crops destroyed in a nationwide agricultural disaster in Nigeria."),
    QuerySpec("Nigeria", "hospital", "This image shows a health-related setting in Nigeria.", "This image shows a hospital emergency caused by a deadly disease outbreak in Nigeria."),
    QuerySpec("South Africa", "market", "This image shows a market or local trade scene in South Africa.", "This image shows emergency food distribution after a nationwide supply collapse in South Africa."),
    QuerySpec("South Africa", "school", "This image shows a school or campus setting in South Africa.", "This image shows schools closing after violent unrest in South Africa."),
    QuerySpec("South Africa", "street", "This image shows an everyday street scene in South Africa.", "This image shows large-scale street violence in South Africa today."),
    QuerySpec("South Africa", "parliament", "This image shows a government or civic setting in South Africa.", "This image shows protests after a disputed election result in South Africa."),
    QuerySpec("South Africa", "farm", "This image shows agricultural activity in South Africa.", "This image shows farmland destroyed during a national food emergency in South Africa."),
    QuerySpec("Rwanda", "market", "This image shows a market scene in Rwanda.", "This image shows emergency shortages in Rwanda after a supply chain collapse."),
    QuerySpec("Rwanda", "school", "This image shows a school setting in Rwanda.", "This image shows students evacuated after a security threat in Rwanda."),
    QuerySpec("Rwanda", "street", "This image shows an everyday street scene in Rwanda.", "This image shows protests and unrest in Rwanda today."),
    QuerySpec("Rwanda", "farm", "This image shows farming or rural work in Rwanda.", "This image shows crop failure caused by a major agricultural disaster in Rwanda."),
    QuerySpec("Rwanda", "hospital", "This image shows a health-related setting in Rwanda.", "This image shows hospitals overwhelmed during a public health emergency in Rwanda."),
    QuerySpec("Uganda", "market", "This image shows a market scene in Uganda.", "This image shows panic buying and shortages in Uganda this week."),
    QuerySpec("Uganda", "school", "This image shows a school or learning setting in Uganda.", "This image shows a school crisis after an attack in Uganda."),
    QuerySpec("Uganda", "street", "This image shows an everyday street scene in Uganda.", "This image shows widespread unrest in Uganda today."),
    QuerySpec("Uganda", "farm", "This image shows agricultural activity in Uganda.", "This image shows a devastating farming crisis in Uganda this season."),
    QuerySpec("Uganda", "hospital", "This image shows a health-related setting in Uganda.", "This image shows hospitals under strain from a deadly outbreak in Uganda."),
    QuerySpec("Ghana", "market", "This image shows a market scene in Ghana.", "This image shows shortages and panic buying in Ghana this week."),
    QuerySpec("Ghana", "school", "This image shows a school setting in Ghana.", "This image shows a school emergency after violent unrest in Ghana."),
    QuerySpec("Ghana", "street", "This image shows an everyday street scene in Ghana.", "This image shows chaos in the streets of Ghana today."),
    QuerySpec("Ghana", "farm", "This image shows farming or agriculture in Ghana.", "This image shows crops destroyed during a major farming disaster in Ghana."),
    QuerySpec("Ghana", "hospital", "This image shows a health-related setting in Ghana.", "This image shows an overwhelmed hospital system in Ghana during an outbreak."),
    QuerySpec("Ethiopia", "market", "This image shows a market scene in Ethiopia.", "This image shows emergency food shortages in Ethiopia this week."),
    QuerySpec("Ethiopia", "school", "This image shows a school or education setting in Ethiopia.", "This image shows schools shutting down after unrest in Ethiopia."),
    QuerySpec("Ethiopia", "street", "This image shows an everyday street scene in Ethiopia.", "This image shows street violence in Ethiopia today."),
    QuerySpec("Ethiopia", "farm", "This image shows farming or agriculture in Ethiopia.", "This image shows severe crop damage during an agricultural emergency in Ethiopia."),
    QuerySpec("Ethiopia", "hospital", "This image shows a health-related setting in Ethiopia.", "This image shows a hospital crisis during a deadly outbreak in Ethiopia."),
    QuerySpec("Senegal", "market", "This image shows a market scene in Senegal.", "This image shows food shortages and public panic in Senegal this week."),
    QuerySpec("Senegal", "school", "This image shows a school setting in Senegal.", "This image shows schools closing after violence in Senegal."),
    QuerySpec("Senegal", "street", "This image shows an everyday street scene in Senegal.", "This image shows unrest in Senegal today."),
    QuerySpec("Senegal", "farm", "This image shows agricultural work in Senegal.", "This image shows a major farming disaster in Senegal this season."),
    QuerySpec("Senegal", "hospital", "This image shows a health-related setting in Senegal.", "This image shows hospitals overwhelmed by an outbreak in Senegal."),
    QuerySpec("Tanzania", "market", "This image shows a market scene in Tanzania.", "This image shows emergency shortages in Tanzania this week."),
    QuerySpec("Tanzania", "school", "This image shows a school setting in Tanzania.", "This image shows a school evacuation after unrest in Tanzania."),
    QuerySpec("Tanzania", "street", "This image shows an everyday street scene in Tanzania.", "This image shows protests spreading across Tanzania today."),
    QuerySpec("Tanzania", "farm", "This image shows farming or agriculture in Tanzania.", "This image shows farms destroyed during a major agricultural crisis in Tanzania."),
    QuerySpec("Tanzania", "hospital", "This image shows a health-related setting in Tanzania.", "This image shows health facilities overwhelmed during an outbreak in Tanzania."),
    QuerySpec("Morocco", "market", "This image shows a market scene in Morocco.", "This image shows panic buying during a supply crisis in Morocco."),
    QuerySpec("Morocco", "school", "This image shows a school or campus setting in Morocco.", "This image shows schools closing after unrest in Morocco."),
    QuerySpec("Morocco", "street", "This image shows an everyday street scene in Morocco.", "This image shows street unrest in Morocco today."),
    QuerySpec("Morocco", "farm", "This image shows agricultural activity in Morocco.", "This image shows farmland ruined during a major farming emergency in Morocco."),
    QuerySpec("Morocco", "hospital", "This image shows a health-related setting in Morocco.", "This image shows a hospital emergency caused by a deadly outbreak in Morocco."),
    QuerySpec("Zimbabwe", "market", "This image shows a market scene in Zimbabwe.", "This image shows shortages and panic buying in Zimbabwe this week."),
    QuerySpec("Zimbabwe", "school", "This image shows a school setting in Zimbabwe.", "This image shows schools shutting down after unrest in Zimbabwe."),
    QuerySpec("Zimbabwe", "street", "This image shows an everyday street scene in Zimbabwe.", "This image shows violent unrest in Zimbabwe today."),
    QuerySpec("Zimbabwe", "farm", "This image shows agricultural work in Zimbabwe.", "This image shows crop losses during a farming disaster in Zimbabwe."),
    QuerySpec("Zimbabwe", "hospital", "This image shows a health-related setting in Zimbabwe.", "This image shows hospitals overwhelmed during a disease emergency in Zimbabwe."),
    QuerySpec("Zambia", "market", "This image shows a market scene in Zambia.", "This image shows emergency shortages in Zambia this week."),
    QuerySpec("Zambia", "school", "This image shows a school setting in Zambia.", "This image shows schools closing after violence in Zambia."),
    QuerySpec("Zambia", "street", "This image shows an everyday street scene in Zambia.", "This image shows unrest in Zambia today."),
    QuerySpec("Zambia", "farm", "This image shows farming or agriculture in Zambia.", "This image shows severe crop damage during an agricultural crisis in Zambia."),
    QuerySpec("Zambia", "hospital", "This image shows a health-related setting in Zambia.", "This image shows hospitals under pressure during an outbreak in Zambia."),
    QuerySpec("Botswana", "market", "This image shows a market scene in Botswana.", "This image shows food shortages and panic buying in Botswana this week."),
    QuerySpec("Botswana", "school", "This image shows a school setting in Botswana.", "This image shows a school shutdown after unrest in Botswana."),
    QuerySpec("Botswana", "street", "This image shows an everyday street scene in Botswana.", "This image shows large protests in Botswana today."),
    QuerySpec("Botswana", "farm", "This image shows agricultural work in Botswana.", "This image shows a farming emergency in Botswana this season."),
    QuerySpec("Botswana", "hospital", "This image shows a health-related setting in Botswana.", "This image shows hospitals overwhelmed during a health crisis in Botswana."),
]


def sanitize_name(name: str) -> str:
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', name)
    return name.strip('_') or 'image'


def fetch_candidates(session: requests.Session, spec: QuerySpec, limit: int = 8) -> list[dict]:
    params = {
        'action': 'query',
        'generator': 'search',
        'gsrsearch': f'{spec.country} {spec.topic} filetype:bitmap',
        'gsrnamespace': 6,
        'gsrlimit': limit,
        'prop': 'imageinfo',
        'iiprop': 'url',
        'format': 'json',
    }
    response = session.get(API_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    pages = list(data.get('query', {}).get('pages', {}).values())
    pages.sort(key=lambda page: page.get('index', 999999))
    return pages


def thumbnail_url(image_url: str, width: int = THUMB_WIDTH) -> str:
    marker = '/wikipedia/commons/'
    if marker not in image_url:
        return image_url
    prefix, suffix = image_url.split(marker, 1)
    filename = suffix.split('/')[-1]
    return f'{prefix}/wikipedia/commons/thumb/{suffix}/{width}px-{quote(filename)}'


def download_image(session: requests.Session, url: str, dest: Path) -> bool:
    response = session.get(url, timeout=30)
    response.raise_for_status()
    content_type = response.headers.get('Content-Type', '')
    if not content_type.startswith('image/'):
        raise RuntimeError(f'Unexpected content type: {content_type}')
    dest.write_bytes(response.content)
    return True


def pick_and_download_images() -> list[dict[str, str]]:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({'User-Agent': USER_AGENT})

    picked: list[dict[str, str]] = []
    seen_remote: set[str] = set()
    candidates_by_spec: list[tuple[QuerySpec, list[dict]]] = []
    for spec in QUERY_SPECS:
        try:
            candidates = fetch_candidates(session, spec)
        except Exception:
            continue
        candidates_by_spec.append((spec, candidates))
        time.sleep(0.2)

    round_index = 0
    while len(picked) < TARGET_IMAGES:
        added_this_round = 0
        for spec, candidates in candidates_by_spec:
            if len(picked) >= TARGET_IMAGES:
                break
            if round_index >= len(candidates):
                continue

            candidate = candidates[round_index]
            info = candidate.get('imageinfo') or []
            if not info:
                continue
            remote_url = info[0].get('url', '').strip()
            if not remote_url or remote_url in seen_remote:
                continue
            if 'upload.wikimedia.org' not in remote_url:
                continue

            ext = Path(remote_url).suffix or '.jpg'
            filename = f"img_{len(picked)+1:04d}{ext}"
            dest = IMAGE_DIR / sanitize_name(filename)
            try:
                download_image(session, thumbnail_url(remote_url), dest)
            except Exception:
                continue

            seen_remote.add(remote_url)
            picked.append({
                'image_path': dest.as_posix(),
                'country_focus': spec.country,
                'language': 'English',
                'likely_text': spec.likely_template,
                'misinformation_text': spec.misinformation_template,
            })
            added_this_round += 1
            time.sleep(0.6)

        if added_this_round == 0:
            break
        round_index += 1

    if len(picked) < TARGET_IMAGES:
        raise RuntimeError(f'Only collected {len(picked)} local images; expected {TARGET_IMAGES}.')

    return picked


def write_dataset(images: list[dict[str, str]]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'image_path', 'text', 'label', 'country_focus', 'language'])
        row_id = 1
        for item in images:
            writer.writerow([row_id, item['image_path'], item['likely_text'], 'likely_consistent', item['country_focus'], item['language']])
            row_id += 1
            writer.writerow([row_id, item['image_path'], item['misinformation_text'], 'misinformation', item['country_focus'], item['language']])
            row_id += 1


def main() -> None:
    images = pick_and_download_images()
    write_dataset(images)
    print(f'Wrote {len(images) * 2} rows to {OUTPUT_PATH}')
    print(f'Saved {len(images)} images to {IMAGE_DIR}')


if __name__ == '__main__':
    main()
