"""Read qualified native identities from the Contacts Receipts v2 contract."""
from __future__ import annotations

import hmac
import json
import os
from http import HTTPStatus
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

import requests


class ContactsLookupError(ValueError):
    def __init__(self, code: str, status: HTTPStatus):
        super().__init__(code)
        self.status = status


def native_id(value):
    try:
        parsed = str(UUID(value))
    except (ValueError, TypeError, AttributeError):
        return None
    return parsed if parsed == value else None


def entity_reference(ref):
    if not isinstance(ref, dict) or ref.get("kind") != "entity":
        return None
    revision = ref.get("revision")
    if not native_id(ref.get("id")) or not isinstance(revision, str) or not revision:
        return None
    return {key: ref[key] for key in ("id", "kind", "revision")}


def identity(value, ref, *, qualification):
    if (
        not ref
        or value.get("entity_kind") not in {"individual", "organization"}
        or value.get("state") not in {"accepted", "provisional"}
        or not isinstance(value.get("name"), str)
    ):
        raise ContactsLookupError("contacts_invalid_identity", HTTPStatus.BAD_GATEWAY)
    return {
        "provider": "contacts-receipts",
        "reference": ref,
        "name": value["name"],
        "kind": value["entity_kind"],
        "state": value["state"],
        "qualification": qualification,
        "speaker_assignment": None,
    }


class ContactsReceiptsLookup:
    def __init__(self, authorization):
        url = os.getenv("TRANSCRIBE_CONTACTS_URL", "")
        self.token = os.getenv("TRANSCRIBE_CONTACTS_TOKEN", "")
        self.perspective = os.getenv("TRANSCRIBE_CONTACTS_PERSPECTIVE", "")
        access_token = os.getenv("TRANSCRIBE_CONTACTS_ACCESS_TOKEN", "")
        if not all((url, self.token, self.perspective, access_token)):
            raise ContactsLookupError("contacts_unconfigured", HTTPStatus.SERVICE_UNAVAILABLE)
        if not isinstance(authorization, str) or not hmac.compare_digest(
            authorization.encode(), ("Bearer " + access_token).encode()
        ):
            raise ContactsLookupError("contacts_forbidden", HTTPStatus.FORBIDDEN)
        try:
            endpoint = urlsplit(url)
            allowed_scheme = endpoint.scheme == "https" or (
                endpoint.scheme == "http" and endpoint.hostname in {"127.0.0.1", "::1", "localhost"}
            )
            if (
                not allowed_scheme or not endpoint.hostname or endpoint.username is not None
                or endpoint.password is not None or endpoint.query or endpoint.fragment
            ):
                raise ValueError("Invalid Contacts endpoint")
        except ValueError:
            raise ContactsLookupError("contacts_unconfigured", HTTPStatus.SERVICE_UNAVAILABLE) from None
        self.endpoint = urlunsplit((endpoint.scheme, endpoint.netloc, "", "", ""))

    def _call(self, operation, payload):
        try:
            with requests.post(
                f"{self.endpoint}/v1/{operation}",
                json={"payload": {"contract_version": 2, **payload}, "perspectives": [self.perspective]},
                headers={"Authorization": "Bearer " + self.token},
                allow_redirects=False, stream=True, timeout=(3, 10),
            ) as response:
                statuses = {
                    401: ("contacts_forbidden", HTTPStatus.FORBIDDEN),
                    403: ("contacts_forbidden", HTTPStatus.FORBIDDEN),
                    404: ("contacts_not_found", HTTPStatus.NOT_FOUND),
                    409: ("contacts_revision_conflict", HTTPStatus.CONFLICT),
                }
                if response.status_code != 200:
                    raise ContactsLookupError(*statuses.get(
                        response.status_code, ("contacts_unavailable", HTTPStatus.SERVICE_UNAVAILABLE)
                    ))
                body = bytearray()
                for chunk in response.iter_content(16384):
                    body.extend(chunk)
                    if len(body) > 1_048_576:
                        raise ContactsLookupError("contacts_response_too_large", HTTPStatus.BAD_GATEWAY)
                value = json.loads(body)
        except requests.RequestException:
            raise ContactsLookupError("contacts_unavailable", HTTPStatus.SERVICE_UNAVAILABLE) from None
        except ContactsLookupError:
            raise
        except ValueError:
            raise ContactsLookupError("contacts_invalid_response", HTTPStatus.BAD_GATEWAY) from None
        if not isinstance(value, dict):
            raise ContactsLookupError("contacts_invalid_response", HTTPStatus.BAD_GATEWAY)
        return value

    def search(self, query):
        query = query.strip()
        if not 1 <= len(query) <= 1000:
            raise ContactsLookupError("contacts_invalid_query", HTTPStatus.BAD_REQUEST)
        value = self._call("resolve", {"query": query})
        items = []
        for group in ("entities", "provisional"):
            entries = value.get(group, [])
            if not isinstance(entries, list):
                raise ContactsLookupError("contacts_invalid_response", HTTPStatus.BAD_GATEWAY)
            for entry in entries:
                if not isinstance(entry, dict) or entry.get("entity_kind") not in {"individual", "organization"}:
                    continue
                refs = entry.get("constituents", [])
                if not isinstance(refs, list):
                    raise ContactsLookupError("contacts_invalid_response", HTTPStatus.BAD_GATEWAY)
                refs = [r for ref in refs if (r := entity_reference(ref))]
                ref = next((r for r in refs if r["id"] == entry.get("entity_id")), None)
                items.append(identity(entry, ref or (refs[0] if len(refs) == 1 else None), qualification="candidate"))
        return {"items": items, "perspective": self.perspective}

    def exact(self, selected_id, revision):
        if not native_id(selected_id):
            raise ContactsLookupError("contacts_invalid_id", HTTPStatus.BAD_REQUEST)
        value = self._call("overview", {"object_id": selected_id})
        ref = entity_reference(value.get("requested_anchor"))
        if not ref or ref["id"] != selected_id:
            raise ContactsLookupError("contacts_identity_conflict", HTTPStatus.CONFLICT)
        if revision is not None and revision != ref["revision"]:
            raise ContactsLookupError("contacts_revision_conflict", HTTPStatus.CONFLICT)
        qualified = value.get("state") == "accepted" and value.get("projection_state") == "current"
        result = identity(value, ref, qualification="exact" if qualified else "candidate")
        return dict(result, perspective=self.perspective)
