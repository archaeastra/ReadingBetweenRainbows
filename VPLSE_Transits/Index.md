# RBR File Tree Example
This file contains part of the RBR data, enough to show off the basic structure as well as how single-atmosphere cases are handled.

The key components are the ``.trn`` and ``.atl`` files, anything else in this filetree will be safely ignored by RBR code.

> [!WARNING]
> Multiple ``.trn`` or ``.atl`` instances in a single end directory will cause errors. Do not do this.

Basic Sructure
- Transit Data:
``./[Star Name]/[Atmosphere Type]/[spec].trn``
- Planet Metadata:
``./[Star Name]/[Atmosphere Type]/[source].atl``
