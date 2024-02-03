# CBOR Ids

The APIs defined in this directory introduce the following CBOR (RFC 8949) ids
used for CBOR Web Token (CWT; RFC 8392) and CBOR Object Signing and Encryption
(COSE; RFC 9052) objects.

## COSE Algorithms

value  | description
------ | ---------------------------------------------------------------
-65537 | HPKE-Base-X25519-SHA256-AES128GCM
-65538 | AEAD_AES_128_GCM_SIV (fixed nonce: x"74DF8FD4BE34AF647F5E54F6")

## CWT Claims

key    | type | description
------ | ---- | ---------------------------------------------
-65537 | bstr | COSE_Key containing the public key parameters
