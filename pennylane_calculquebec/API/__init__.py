"""Network and API layer for communicating with the MonarQ quantum computer.

Contains:

* :class:`~pennylane_calculquebec.API.adapter.ApiAdapter` — singleton that wraps all Thunderhead REST calls
* :class:`~pennylane_calculquebec.API.client.ApiClient` — client credentials and configuration
* :class:`~pennylane_calculquebec.API.job.Job` — job submission and polling logic
* :func:`~pennylane_calculquebec.API.retry_decorator.retry` — exponential-backoff retry decorator
"""

from pennylane_calculquebec.exceptions import ApiError
