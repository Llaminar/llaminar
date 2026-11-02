/** @file HostServiceTLSControl.cpp
 * @brief Real relocated TLS-pointer provider for portable identity and loader-lifetime controls.
 */
extern "C" {
int host_service_tls_anchor=7;
thread_local int* host_service_tls_pointer=&host_service_tls_anchor;
/** @return Actual TLS pointer, forcing materialization in callers that inspect it. */
int* host_service_tls_address(){return host_service_tls_pointer;}
}
